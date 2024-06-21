#pragma once
#include <complex>

#include "absl/types/span.h"
#include "seal/util/croots.h"
#include "seal/util/dwthandler.h"

namespace seal::util {

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
      auto c = a * s;                                                   \
      return c;                                                         \
    }                                                                   \
    inline std::complex<TT> mul_root_scalar(const std::complex<TT>& r,  \
                                            const TT& s) const {        \
      return r * s;                                                     \
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

  std::vector<scalar_t> fwd_mat_real_;
  std::vector<scalar_t> fwd_mat_imag_;
  std::vector<scalar_t> bwd_mat_real_;
  std::vector<scalar_t> bwd_mat_imag_;

  void InitFFTMatrix(int n) {
    fwd_mat_real_.resize(n * n);
    fwd_mat_imag_.resize(n * n);
    bwd_mat_real_.resize(n * n);
    bwd_mat_imag_.resize(n * n);

    for (int j = 0; j < n; ++j) {
      // omega^{j * k}
      for (int k = 0; k < n; ++k) {
        auto powers = std::polar(1.0, -2.0 * j * k * M_PI / n);
        auto inv_powers = std::polar(1.0, 2.0 * j * k * M_PI / n);

        fwd_mat_real_[j * n + k] = EncodeToFxp<scalar_t>(powers.real(), fxp_);
        fwd_mat_imag_[j * n + k] = EncodeToFxp<scalar_t>(powers.imag(), fxp_);

        bwd_mat_real_[j * n + k] =
            EncodeToFxp<scalar_t>(inv_powers.real() / n, fxp_);
        bwd_mat_imag_[j * n + k] =
            EncodeToFxp<scalar_t>(inv_powers.imag() / n, fxp_);
      }
    }
  }

  MPCCKKSEncoder(int fxp, size_t degree) : fxp_(fxp) {
    SPU_ENFORCE(absl::has_single_bit(degree));
    SPU_ENFORCE(degree > 4);
    slots_ = degree / 2;
    int logn = std::log2(degree);

    // InitFFTMatrix(slots_);

    matrix_reps_index_map_.resize(degree);

    // Copy from the matrix to the value vectors
    uint64_t gen = 3;
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

    c64_root_powers_.resize(degree);
    c64_inv_root_powers_.resize(degree);

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

      c64_root_powers_[i] = rp;
      c64_inv_root_powers_[i] = inv_rp;
    }

    mpc_arith_ = MPCArith(fxp_);
    fft_handler_ = FFTHandler(mpc_arith_);

    c64_arith_ = ComplexArith();
    c64_fft_handler_ = C64_FFTHandler(c64_arith_);
  }

  void encode(absl::Span<const double> input,
              absl::Span<double> destination) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), slots_);
    SPU_ENFORCE_EQ(destination.size(), n);

    auto conj_values = seal::util::allocate<std::complex<double>>(
        n, seal::MemoryManager::GetPool(), 0);

    for (size_t i = 0; i < input.size(); i++) {
      conj_values[matrix_reps_index_map_[i]] = input[i];

      conj_values[matrix_reps_index_map_[i + slots_]] = std::conj(input[i]);
    }

    double fix = 1. / static_cast<double>(n);
    c64_fft_handler_.transform_from_rev(conj_values.get(),
                                        seal::util::get_power_of_two(n),
                                        c64_inv_root_powers_.data(), &fix);
    std::transform(conj_values.get(), conj_values.get() + n, destination.data(),
                   [](const auto& x) -> double { return x.real(); });
  }

  void encode(absl::Span<const scalar_t> input,
              absl::Span<scalar_t> destination) const {
    const size_t n = slots_ * 2;
    SPU_ENFORCE_EQ(input.size(), slots_);
    SPU_ENFORCE_EQ(destination.size(), n);
    // (V + U*j) * x
    // V*x + U*x*j
    for (size_t i = 0; i < slots_; ++i) {
      scalar_t real_accum = 0;
      scalar_t imag_accum = 0;
      for (size_t j = 0; j < slots_; ++j) {
        real_accum += bwd_mat_real_.at(i * slots_ + j) * input[j];
        imag_accum += bwd_mat_imag_.at(i * slots_ + j) * input[j];
      }
      destination[i] = real_accum;
      destination[slots_ + i] = imag_accum;
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
    // for (size_t i = 0; i < slots_; ++i) {
    //   scalar_t real_accum = 0;
    //   scalar_t imag_accum = 0;
    //   for (size_t j = 0; j < slots_; ++j) {
    //     real_accum += fwd_mat_real_.at(i * slots_ + j) * input[j];
    //     imag_accum += fwd_mat_imag_.at(i * slots_ + j) * input[slots_ + j];
    //   }
    //   destination[i] = real_accum - imag_accum;
    // }
  }

  inline size_t slot_count() const noexcept { return slots_; }

 private:
  int fxp_;
  size_t slots_;
  std::vector<std::size_t> matrix_reps_index_map_;

  std::vector<root_t> root_powers_;
  std::vector<root_t> inv_root_powers_;
  MPCArith mpc_arith_;
  FFTHandler fft_handler_;

  std::vector<std::complex<double>> c64_root_powers_;
  std::vector<std::complex<double>> c64_inv_root_powers_;
  ComplexArith c64_arith_;
  C64_FFTHandler c64_fft_handler_;
};
