#pragma once
#include <complex>

#include "absl/types/span.h"
#include "seal/seal.h"

class FastCKKSEncoder {
 public:
  using F64 = double;
  using C64 = std::complex<double>;
  using C64Vec = std::vector<C64>;

  explicit FastCKKSEncoder(std::shared_ptr<seal::SEALContext> context)
      : context_(context) {
    auto& context_data = *context_->first_context_data();
    degree_ = context_data.parms().poly_modulus_degree();
    max_nslots_ = degree_ >> 1;
    logn_ = seal::util::get_power_of_two(degree_);
    m_ = degree_ << 1;

    rotGroup_ = RotGroupHelper::get(degree_, degree_ >> 1);

    const F64 angle = 2 * M_PI / m_;
    roots_.resize(m_ + 1);
    for (size_t j = 0; j < m_; ++j) {
      roots_[j] = std::polar<F64>(1.0, angle * j);
    }
    roots_[m_] = roots_[0];
  }

  template <typename T>
  void Encode(const std::vector<T>& vec, const F64 scale,
              seal::Plaintext* out) {
    CHECK_BOOL(out == nullptr, Status::ArgumentError("Encode: nULL pointer"));

    if (out->parms_id() == seal::parms_id_zero) {
      out->parms_id() = context_->first_parms_id();
    }

    const size_t input_size = vec.size();
    CHECK_BOOL(!IsValidLength(input_size),
               Status::ArgumentError("Invalid length to pack"));

    auto mem_pool = seal::MemoryManager::GetPool();
    auto conj_values = seal::util::allocate<C64>(input_size, mem_pool);

    std::transform(vec.cbegin(), vec.cend(), conj_values.get(),
                   [](T const& v) -> C64 { return static_cast<C64>(v); });

    Pack(conj_values.get(), input_size);  // invFFT

    F64* r_start = reinterpret_cast<F64*>(conj_values.get());
    F64* r_end = r_start + input_size * 2;
    F64 sn = scale / input_size;
    std::transform(r_start, r_end, r_start,
                   [sn](F64 v) -> F64 { return v * sn; });

    int total_nbits = context_->get_context_data(out->parms_id())
                          ->total_coeff_modulus_bit_count();
    F64 upper_limit = static_cast<F64>(1UL << std::min(62, total_nbits));
    bool any_large = std::any_of(r_start, r_end, [upper_limit](F64 v) {
      return std::fabs(v) >= upper_limit;
    });
    CHECK_BOOL(any_large, Status::ArgumentError("Encode: scale out of bound"));

    auto coeffients = seal::util::allocate<I64>(degree_, mem_pool);
    RoundCoeffients(conj_values.get(), input_size, coeffients.get());
    CHECK_BOOL(!ApplyNTT(coeffients.get(), degree_, out),
               Status::InternalError("ApplyNTT error"));
    out->scale() = scale;

    return Status::Ok();
  }

  Status Decode(Impl& rt, const Ptx& in, size_t length, C64Vec* out) {
    using namespace seal;
    auto context_data_ptr = context_->get_context_data(in.parms_id());
    if (!context_data_ptr) {
      return Status::ArgumentError("invalid plaintext");
    }

    const size_t gap = max_nslots_ / length;
    std::vector<F64> coeffients;
    CHECK_STATUS(rt.ConventionalForm(in, &coeffients, true, gap));

    if (coeffients.size() != length * 2) {
      throw std::length_error("Decode fail");
    }

    out->resize(length);
    for (size_t i = 0; i < length; ++i) {
      out->at(i).real(coeffients.at(i));
      out->at(i).imag(coeffients.at(length + i));
    }

    Unpack(out->data(), length);
    return Status::Ok();
  }

  void RoundCoeffients(const C64* array, size_t nslots, I64* dst) const {
    if (!array || !IsValidLength(nslots)) {
      throw std::invalid_argument("RoundCoeffients: invalid_argument");
    }

    const int gap = max_nslots_ / nslots;
    if (gap != 1) {
      std::fill(dst, dst + degree_, 0UL);
    }

    I64* real_part = dst;
    I64* imag_part = dst + max_nslots_;

    for (size_t i = 0; i < nslots; ++i, real_part += gap, imag_part += gap) {
      *real_part = static_cast<I64>(std::round(array[i].real()));
      *imag_part = static_cast<I64>(std::round(array[i].imag()));
    }
  }

  bool ApplyNTT(const I64* coeffients, size_t length, Ptx* out) const {
    if (!out) {
      return false;
    }
    const auto pid = out->parms_id();
    const auto context_data = context_->get_context_data(pid);
    const size_t nmoduli = context_data->parms().coeff_modulus().size();
    const auto& small_ntt_tables = context_data->small_ntt_tables();
    if (!coeffients || length != degree_ || pid == seal::parms_id_zero) {
      return false;
    }

    out->parms_id() = seal::parms_id_zero;  // stop the warning in resize()
    out->resize(length * nmoduli);

    U64* dst_ptr = out->data();
    for (size_t cm = 0; cm < nmoduli; ++cm, dst_ptr += degree_) {
      const auto& modulus = small_ntt_tables[cm];  //.modulus();
      const U64 p = modulus.modulus().value();
      std::transform(coeffients, coeffients + length, dst_ptr,
                     [&modulus, p](I64 v) -> U64 {
                       bool sign = v < 0;
                       U64 vv = seal::util::barrett_reduce_64(
                           (U64)std::abs(v), modulus.modulus());
                       return vv > 0 ? (sign ? p - vv : vv) : 0;
                     });

      seal::util::ntt_negacyclic_harvey(dst_ptr, modulus);
    }
    out->parms_id() = pid;
    return true;
  }

  inline bool IsValidLength(size_t len) const {
    return !(len <= 0 || len > max_nslots_ || max_nslots_ % len != 0);
  }

  inline C64 C64Mul(C64 const& a, const C64& b) const {
    F64 x = a.real(), y = a.imag(), u = b.real(), v = b.imag();
    return C64(x * u - y * v, x * v + y * u);
  }

  inline void Invbutterfly(C64* x0, C64* x1, C64 const& w) const {
    C64 u = *x0;
    C64 v = *x1;
    *x0 = u + v;
    *x1 = C64Mul(u - v, w);
  }

  inline void Butterfly(C64* x0, C64* x1, C64 const& w) const {
    C64 u = *x0;
    C64 v = C64Mul(*x1, w);
    *x0 = u + v;
    *x1 = u - v;
  }

  void Pack(C64* vals, size_t n) const {
    if (!vals || !IsValidLength(n)) {
      throw std::invalid_argument("pack invalid argument");
    }

    for (size_t len = n / 2, h = 1; len > 2; len >>= 1, h <<= 1) {
      const size_t quad = len << 3;
      const size_t gap = m_ / quad;
      C64* x0 = vals;
      C64* x1 = x0 + len;
      for (size_t i = 0; i < h; ++i) {
        const size_t* rot = rotGroup_.data();
        size_t idx;
        for (size_t j = 0; j < len; j += 4) {
          idx = (quad - (*rot++ & (quad - 1))) * gap;
          Invbutterfly(x0++, x1++, roots_[idx]);

          idx = (quad - (*rot++ & (quad - 1))) * gap;
          Invbutterfly(x0++, x1++, roots_[idx]);

          idx = (quad - (*rot++ & (quad - 1))) * gap;
          Invbutterfly(x0++, x1++, roots_[idx]);

          idx = (quad - (*rot++ & (quad - 1))) * gap;
          Invbutterfly(x0++, x1++, roots_[idx]);
        }

        x0 += len;
        x1 += len;
      }
    }  // main loop

    {  // len = 2, h = n / 4, quad = 16
      C64* x0 = vals;
      C64* x1 = x0 + 2;

      const size_t idx0 = (16 - (rotGroup_[0] & 15)) * (m_ / 16);
      const size_t idx1 = (16 - (rotGroup_[1] & 15)) * (m_ / 16);
      for (size_t i = 0; i < n / 4; ++i) {
        Invbutterfly(x0++, x1++, roots_[idx0]);
        Invbutterfly(x0, x1, roots_[idx1]);
        x0 += 3;
        x1 += 3;
      }
    }

    {  // len = 1
      C64* x0 = vals;
      C64* x1 = x0 + 1;

      const long idx = (8 - (rotGroup_[0] & 7)) * (m_ / 8);
      for (size_t i = 0; i < n / 2; ++i) {
        Invbutterfly(x0, x1, roots_[idx]);
        x0 += 2;
        x1 += 2;
      }
    }

    RevBinPermute(vals, n);
  }

  void Unpack(C64* vals, size_t n) const {
    if (!vals || !IsValidLength(n)) {
      throw std::invalid_argument("unpack invalid argument");
    }

    RevBinPermute(vals, n);

    {  // len = 1, h = n / 2, quad = 8
      C64* x0 = vals;
      C64* x1 = x0 + 1;

      const long idx = (rotGroup_[0] & 7) * (m_ / 8);
      for (size_t i = 0; i < n / 2; ++i) {
        Butterfly(x0, x1, roots_[idx]);
        x0 += 2;
        x1 += 2;
      }
    }

    {  // len = 2, h = n / 4, quad = 16
      C64* x0 = vals;
      C64* x1 = x0 + 2;

      const size_t idx0 = (rotGroup_[0] & 15) * (m_ / 16);
      const size_t idx1 = (rotGroup_[1] & 15) * (m_ / 16);
      for (size_t i = 0; i < n / 4; ++i) {
        Butterfly(x0++, x1++, roots_[idx0]);
        Butterfly(x0, x1, roots_[idx1]);
        x0 += 3;
        x1 += 3;
      }
    }

    for (size_t len = 4, h = n / 8; len < n; len <<= 1, h >>= 1) {
      const long quad = (len << 3) - 1;  // mod 8 * len
      const long gap = m_ / (quad + 1);

      C64* x0 = vals;
      C64* x1 = x0 + len;

      for (size_t i = 0; i < h; ++i) {
        const size_t* rot = rotGroup_.data();
        long idx;
        for (size_t j = 0; j < len; j += 4) {
          idx = ((*rot++ & quad)) * gap;
          Butterfly(x0++, x1++, roots_[idx]);

          idx = ((*rot++ & quad)) * gap;
          Butterfly(x0++, x1++, roots_[idx]);

          idx = ((*rot++ & quad)) * gap;
          Butterfly(x0++, x1++, roots_[idx]);

          idx = ((*rot++ & quad)) * gap;
          Butterfly(x0++, x1++, roots_[idx]);
        }
        x0 += len;
        x1 += len;
      }
    }
  }

  template <typename T>
  void RevBinPermute(T* array, size_t length) const {
    if (length <= 2) return;
    for (size_t i = 1, j = 0; i < length; ++i) {
      size_t bit = length >> 1;
      for (; j >= bit; bit >>= 1) {
        j -= bit;
      }
      j += bit;

      if (i < j) {
        std::swap(array[i], array[j]);
      }
    }
  }

  C64Vec roots_;
  std::vector<size_t> rotGroup_;
  size_t max_nslots_, logn_, degree_, m_;
  std::shared_ptr<seal::SEALContext> context_{nullptr};
};
