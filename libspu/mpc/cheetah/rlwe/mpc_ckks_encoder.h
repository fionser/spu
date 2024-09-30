#pragma once
#include <complex>

#include "Eigen/Sparse"
#include "absl/types/span.h"
#include "seal/util/croots.h"
#include "seal/util/dwthandler.h"

using C64 = std::complex<double>;
using C64Mat = Eigen::SparseMatrix<C64>;
namespace seal::util {

// NOTE(lwj): we perform local truncation for fixed-point mul during
// the FFT matrix evaluation.
#define DEF_ARITH(TT)                                                   \
  template <>                                                           \
  class Arithmetic<std::complex<TT>, std::complex<TT>, TT> {            \
    int fxp_;                                                           \
                                                                        \
   public:                                                              \
    explicit Arithmetic(int fxp = 0) : fxp_(fxp) {}                     \
    inline std::complex<TT> add(const std::complex<TT>& a,              \
                                const std::complex<TT>& b) const {      \
      return a + b;                                                     \
    }                                                                   \
    inline std::complex<TT> sub(const std::complex<TT>& a,              \
                                const std::complex<TT>& b) const {      \
      return a - b;                                                     \
    }                                                                   \
    inline std::complex<TT> mul_root(const std::complex<TT>& a,         \
                                     const std::complex<TT>& r) const { \
      using ST = std::make_signed<TT>::type;                            \
      auto c = a * r;                                                   \
      c.real(static_cast<ST>(c.real()) >> fxp_);                        \
      c.imag(static_cast<ST>(c.imag()) >> fxp_);                        \
      return c;                                                         \
    }                                                                   \
    inline std::complex<TT> mul_scalar(const std::complex<TT>& a,       \
                                       const TT& s) const {             \
      using ST = std::make_signed<TT>::type;                            \
      auto c = a * s;                                                   \
      c.real(static_cast<ST>(c.real()) >> fxp_);                        \
      c.imag(static_cast<ST>(c.imag()) >> fxp_);                        \
      return c;                                                         \
    }                                                                   \
    inline std::complex<TT> mul_root_scalar(const std::complex<TT>& r,  \
                                            const TT& s) const {        \
      using ST = std::make_signed<TT>::type;                            \
      auto c = r * s;                                                   \
      c.real(static_cast<ST>(c.real()) >> fxp_);                        \
      c.imag(static_cast<ST>(c.imag()) >> fxp_);                        \
      return c;                                                         \
    }                                                                   \
    inline std::complex<TT> guard(const std::complex<TT>& a) const {    \
      return a;                                                         \
    }                                                                   \
  }

DEF_ARITH(uint32_t);
DEF_ARITH(uint64_t);
DEF_ARITH(uint128_t);

}  // namespace seal::util

template <typename T>
T EncodeToFxp(double x, int fxp) {
  T u = std::roundf(std::abs(x) * static_cast<double>(1L << fxp));
  if (std::signbit(x)) {
    return -u;
  }
  return u;
}

template <typename T>
double DecodeFromFxp(T x, int fxp) {
  using S = typename std::make_signed<T>::type;
  return static_cast<S>(x) / static_cast<double>(1L << fxp);
}

template <typename scalar_ty>
class MPCCKKSEncoder {
 public:
  using scalar_t = scalar_ty;
  using value_t = std::complex<scalar_t>;
  using root_t = std::complex<scalar_t>;

  using MPCArith = seal::util::Arithmetic<value_t, root_t, scalar_t>;
  using FFTHandler = seal::util::DWTHandler<value_t, root_t, scalar_t>;

  using ComplexArith = seal::util::Arithmetic<std::complex<double>,
                                              std::complex<double>, double>;
  using C64_FFTHandler = seal::util::DWTHandler<std::complex<double>,
                                                std::complex<double>, double>;

  std::complex<double> exp_c64(std::complex<double> base, uint64_t exp) {
    std::complex<double> result = 1;

    while (exp > 0) {
      // If exp is odd, multiply the current result by base
      if (exp % 2 == 1) {
        result = result * base;
      }
      base = base * base;
      // Divide exp by 2
      exp >>= 1;
    }
    return result;
  }

  MPCCKKSEncoder(int fxp, size_t degree) : fxp_(fxp) {
    SPU_ENFORCE(absl::has_single_bit(degree));
    SPU_ENFORCE(degree > 4);
    slots_ = degree / 2;
    int logn = std::log2(degree);

    matrix_reps_index_map_.resize(degree);

    // NOTE(lwj): change SEAL's generator to 5.
    uint64_t gen = 5;
    uint64_t pos = 1;
    uint64_t m = static_cast<uint64_t>(degree) << 1;
    for (size_t i = 0; i < slots_; i++) {
      // Position in normal bit order
      uint64_t index1 = (pos - 1) >> 1;
      uint64_t index2 = (m - pos - 1) >> 1;

      // Set the bit-reversed locations
      matrix_reps_index_map_[i] =
          seal::util::safe_cast<size_t>(seal::util::reverse_bits(index1, logn));

      matrix_reps_index_map_[slots_ | i] =
          seal::util::safe_cast<size_t>(seal::util::reverse_bits(index2, logn));

      // Next primitive root
      pos *= gen;
      pos &= (m - 1);
    }

    // We need 1~(n-1)-th powers of the primitive 2n-th root, m = 2n
    root_powers_.resize(degree);
    inv_root_powers_.resize(degree);

    seal::util::ComplexRoots complex_roots(static_cast<size_t>(m),
                                           seal::MemoryManager::GetPool());

    for (size_t i = 1; i < degree; i++) {
      auto rp = complex_roots.get_root(seal::util::reverse_bits(i, logn));

      root_powers_[i].real(EncodeToFxp<scalar_t>(rp.real(), fxp_));
      root_powers_[i].imag(EncodeToFxp<scalar_t>(rp.imag(), fxp_));

      auto inv_rp = std::conj(
          complex_roots.get_root(seal::util::reverse_bits(i - 1, logn) + 1));

      inv_root_powers_[i].real(EncodeToFxp<scalar_t>(inv_rp.real(), fxp_));
      inv_root_powers_[i].imag(EncodeToFxp<scalar_t>(inv_rp.imag(), fxp_));
    }

    mpc_arith_ = MPCArith(fxp_);
    fft_handler_ = FFTHandler(mpc_arith_);

    decode_mat_.resize(slots_ * degree);
    encode_mat_.resize(degree * slots_);

    uint64_t gen_pow = 1;
    // V[r, c] = w^{5^r * c}
    // invV = 1/N * conj(V^T)
    for (size_t r = 0; r < slots_; r++) {
      for (size_t c = 0; c < degree; c++) {
        // V[r, c] = w^{5^r * c}
        auto v = complex_roots.get_root((gen_pow * c) & (m - 1));
        decode_mat_[r * degree + c].real(EncodeToFxp<scalar_t>(v.real(), fxp_));
        decode_mat_[r * degree + c].imag(EncodeToFxp<scalar_t>(v.imag(), fxp_));

        // transpose V^T
        encode_mat_[c * slots_ + r].real(EncodeToFxp<scalar_t>(v.real(), fxp_));
        encode_mat_[c * slots_ + r].imag(EncodeToFxp<scalar_t>(v.imag(), fxp_));
      }

      gen_pow *= gen;
      gen_pow &= (m - 1);
    }

#if 0
    // To test V * invV is identity matrix
    for (size_t i = 0; i < slots_; i++) {
      for (size_t j = 0; j < slots_; j++) {
        root_t tmp0 = 0;
        for (size_t k = 0; k < degree; k++) {
          tmp0 += decode_mat_[i * degree + k] * encode_mat_[k * slots_ + j];
        }
        printf("%f + %f ", DecodeFromFxp<scalar_t>(tmp0.real(), 2 * fxp_),
               DecodeFromFxp<scalar_t>(tmp0.imag(), 2 * fxp_));
      }
      printf("\n");
    }
#endif
  }

  // FIXME(lwj): encode_complex_new is not mixing with encode_complex
  void encode_complex_new(absl::Span<const value_t> input,
                          absl::Span<scalar_t> destination) const {
    const size_t n = slots_ * 2;
    size_t logn = std::log2(n);
    SPU_ENFORCE_EQ(input.size(), slots_);
    SPU_ENFORCE_EQ(destination.size(), n);

    // Ref https://eprint.iacr.org/2018/153.pdf
    // Section 4.1 Packing Method
    using ST = typename std::make_signed<scalar_t>::type;
    for (size_t r = 0; r < n; r++) {
      scalar_t tmp = 0;
      // By definition, m(X) = V * conj(v) + conj(V) * v
      for (size_t c = 0; c < slots_; c++) {
        tmp += 2 * encode_mat_[r * slots_ + c].real() * input[c].real();
        tmp += 2 * encode_mat_[r * slots_ + c].imag() * input[c].imag();
      }

      destination[r] =
          static_cast<scalar_t>(static_cast<ST>(tmp) >> (fxp_ + logn));
    }
  }

  void encode_complex(absl::Span<const value_t> input,
                      absl::Span<scalar_t> destination) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), slots_);
    SPU_ENFORCE_EQ(destination.size(), n);

    auto conj =
        seal::util::allocate<value_t>(n, seal::MemoryManager::GetPool(), 0);

    for (size_t i = 0; i < slots_; ++i) {
      conj[matrix_reps_index_map_[i]] = input[i];
      conj[matrix_reps_index_map_[i + slots_]] = std::conj(input[i]);
    }

    scalar_t fix = EncodeToFxp<scalar_t>(1. / n, fxp_);
    fft_handler_.transform_from_rev(conj.get(), seal::util::get_power_of_two(n),
                                    inv_root_powers_.data(), &fix);
    for (size_t i = 0; i < n; ++i) {
      destination[i] = conj[i].real();
    }
  }

  void encode(absl::Span<const scalar_t> input,
              absl::Span<scalar_t> destination) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), slots_);
    SPU_ENFORCE_EQ(destination.size(), n);

    auto conj =
        seal::util::allocate<value_t>(n, seal::MemoryManager::GetPool(), 0);

    for (size_t i = 0; i < slots_; ++i) {
      conj[matrix_reps_index_map_[i]].real(input[i]);
      // NOTE(optimize)
      // conj[matrix_reps_index_map_[i + slots_]].real(input[i]);
    }

    scalar_t fix = EncodeToFxp<scalar_t>(1. / n, fxp_);
    fft_handler_.transform_from_rev(conj.get(), seal::util::get_power_of_two(n),
                                    inv_root_powers_.data(), &fix);
    for (size_t i = 0; i < n; ++i) {
      destination[i] = 2 * conj[i].real();
    }
  }

  void decode_complex(absl::Span<const scalar_t> input,
                      absl::Span<scalar_t> real_dest,
                      absl::Span<scalar_t> imag_dest) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), n);
    SPU_ENFORCE_EQ(real_dest.size(), slots_);
    SPU_ENFORCE_EQ(imag_dest.size(), slots_);

    auto res =
        seal::util::allocate<value_t>(n, seal::MemoryManager::GetPool(), 0);
    for (size_t i = 0; i < n; ++i) {
      res[i].real(input[i]);
    }

    fft_handler_.transform_to_rev(res.get(), seal::util::get_power_of_two(n),
                                  root_powers_.data());
    for (size_t i = 0; i < slots_; i++) {
      real_dest[i] = res[matrix_reps_index_map_[i]].real();
      auto t = res[matrix_reps_index_map_[i]].imag();
      imag_dest[i] = t;  // NOTE(lwj): not sure why this work
    }
  }

  void decode_complex_new(absl::Span<const scalar_t> input,
                          absl::Span<scalar_t> real_dest,
                          absl::Span<scalar_t> imag_dest) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), n);
    SPU_ENFORCE_EQ(real_dest.size(), slots_);
    SPU_ENFORCE_EQ(imag_dest.size(), slots_);
    using ST = typename std::make_signed<scalar_t>::type;
    // V * input
    for (size_t r = 0; r < slots_; ++r) {
      root_t tmp = 0;
      for (size_t c = 0; c < n; ++c) {
        tmp += decode_mat_[r * n + c] * input[c];
      }
      real_dest[r] =
          static_cast<scalar_t>(static_cast<ST>(tmp.real()) >> (fxp_));
      imag_dest[r] =
          static_cast<scalar_t>(static_cast<ST>(tmp.imag()) >> (fxp_));
    }
  }

  void decode(absl::Span<const scalar_t> input,
              absl::Span<scalar_t> destination) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), n);
    SPU_ENFORCE_EQ(destination.size(), slots_);

    auto res =
        seal::util::allocate<value_t>(n, seal::MemoryManager::GetPool(), 0);
    for (size_t i = 0; i < n; ++i) {
      res[i].real(input[i]);
    }

    fft_handler_.transform_to_rev(res.get(), seal::util::get_power_of_two(n),
                                  root_powers_.data());
    for (size_t i = 0; i < slots_; i++) {
      destination[i] = res[matrix_reps_index_map_[i]].real();
    }
  }

  inline size_t slot_count() const noexcept { return slots_; }

 private:
  int fxp_;
  size_t slots_;
  std::vector<std::size_t> matrix_reps_index_map_;

  std::vector<root_t> encode_mat_;
  std::vector<root_t> decode_mat_;

  std::vector<root_t> root_powers_;
  std::vector<root_t> inv_root_powers_;
  MPCArith mpc_arith_;
  FFTHandler fft_handler_;
};
