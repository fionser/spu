#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <sstream>

#include "seal/seal.h"
#include "seal/util/dwthandler.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/uintcore.h"
#include "utils.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/mpc_ckks_encoder.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

using namespace spu::mpc::cheetah;

void NegacyclicRightShiftInplace(seal::Ciphertext& ct, size_t shift,
                                 const seal::SEALContext& context) {
  if (shift == 0 || ct.size() == 0) {
    // nothing to do
    return;
  }

  auto cntxt = context.get_context_data(ct.parms_id());
  SPU_ENFORCE(cntxt != nullptr, "invalid ct");
  SPU_ENFORCE(not ct.is_ntt_form(), "need non-ntt ct for negacyclic shift");

  size_t num_coeff = ct.poly_modulus_degree();
  SPU_ENFORCE(shift < num_coeff);

  std::vector<uint64_t> tmp(shift);
  //  i < N - s  ai*X^i -> ai*X^{i + s}
  // i >= N - s ai*X^i -> -ai*X^{(i + s) % N}
  const auto& modulus = cntxt->parms().coeff_modulus();
  for (size_t k = 0; k < ct.size(); ++k) {
    uint64_t* dst_ptr = ct.data(k);

    for (const auto& prime : modulus) {
      // save [N-s, N)
      std::copy_n(dst_ptr + num_coeff - shift, shift, tmp.data());

      // X^i for i \in [0, N-s)
      for (size_t i = num_coeff - shift; i > 0; --i) {
        dst_ptr[i - 1 + shift] = dst_ptr[i - 1];
      }

      // i \n [N-s, N)
      for (size_t i = 0; i < shift; ++i) {
        dst_ptr[i] = seal::util::negate_uint_mod(tmp[i], prime);
      }

      dst_ptr += num_coeff;
    }
  }
}

void MulImageUnitInplace(seal::Ciphertext& ct,
                         const seal::SEALContext& context) {
  const bool is_ntt = ct.is_ntt_form();
  const size_t degree = ct.poly_modulus_degree();
  seal::Evaluator evaluator(context);
  if (is_ntt) {
    evaluator.transform_from_ntt_inplace(ct);
  }

  // * X^{N/2}
  NegacyclicRightShiftInplace(ct, degree / 2, context);

  if (is_ntt) {
    evaluator.transform_to_ntt_inplace(ct);
  }
}

void ToCKKSPt(absl::Span<const double> coeff, double scale,
              const seal::SEALContext& context, seal::Plaintext& pt) {
  const auto cntx = context.first_context_data();
  const auto& params = cntx->parms();
  const auto& modulus = params.coeff_modulus();

  size_t n = coeff.size();
  size_t N = params.poly_modulus_degree();
  SPU_ENFORCE(n > 0 and n <= N);

  pt.parms_id() = seal::parms_id_zero;
  pt.resize(N * modulus.size());
  for (size_t j = 0; j < modulus.size(); ++j) {
    for (size_t i = 0; i < n; ++i) {
      uint64_t v = std::round(scale * std::abs(coeff[i]));
      v = seal::util::barrett_reduce_64(v, modulus[j]);
      if (coeff[i] < 0) {
        v = seal::util::negate_uint_mod(v, modulus[j]);
      }
      pt[j * N + i] = v;
    }
  }

  pt.parms_id() = cntx->parms_id();
}

void FromCKKSPt(const seal::Plaintext& pt, double scale,
                const seal::SEALContext& context, absl::Span<double> out) {
  const auto cntx = context.get_context_data(pt.parms_id());
  SPU_ENFORCE(cntx != nullptr);
  const auto& params = cntx->parms();
  const auto& modulus = params.coeff_modulus();

  size_t N = params.poly_modulus_degree();
  size_t L = modulus.size();
  SPU_ENFORCE_EQ(out.size(), N);

  std::vector<uint64_t> plain_copy(pt.coeff_count());
  std::copy_n(pt.data(), plain_copy.size(), plain_copy.data());
  cntx->rns_tool()->base_q()->compose_array(plain_copy.data(), N,
                                            seal::MemoryManager::GetPool());
  auto decryption_modulus = cntx->total_coeff_modulus();
  auto upper_half_threshold = cntx->upper_half_threshold();

  // Create floating-point representations of the multi-precision integer
  // coefficients
  double two_pow_64 = std::pow(2.0, 64);

  for (size_t i = 0; i < N; i++) {
    out[i] = 0.0;
    if (seal::util::is_greater_than_or_equal_uint(plain_copy.data() + (i * L),
                                                  upper_half_threshold, L)) {
      double scaled_two_pow_64 = 1.0 / scale;
      for (std::size_t j = 0; j < L; j++, scaled_two_pow_64 *= two_pow_64) {
        if (plain_copy[i * L + j] > decryption_modulus[j]) {
          auto diff = plain_copy[i * L + j] - decryption_modulus[j];
          out[i] += diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
        } else {
          auto diff = decryption_modulus[j] - plain_copy[i * L + j];
          out[i] -= diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
        }
      }
    } else {
      double scaled_two_pow_64 = 1.0 / scale;
      for (std::size_t j = 0; j < L; j++, scaled_two_pow_64 *= two_pow_64) {
        auto curr_coeff = plain_copy[i * L + j];
        out[i] += curr_coeff
                      ? static_cast<double>(curr_coeff) * scaled_two_pow_64
                      : 0.0;
      }
    }
  }
}

void AddInplace(seal::Plaintext& pt, const seal::Plaintext& oth,
                const seal::SEALContext& context) {
  SPU_ENFORCE_EQ(pt.coeff_count(), oth.coeff_count());

  const auto cntx = context.get_context_data(pt.parms_id());
  SPU_ENFORCE(cntx != nullptr);
  const auto& params = cntx->parms();
  const auto& modulus = params.coeff_modulus();

  size_t N = params.poly_modulus_degree();
  size_t L = modulus.size();
  for (size_t j = 0; j < L; ++j) {
    seal::util::add_poly_coeffmod(pt.data() + j * N, oth.data() + j * N, N,
                                  modulus[j], pt.data() + j * N);
  }
}

void add_uint_mod_inplace(uint64_t* op0, const uint64_t* op1,
                          const uint64_t* modulus, size_t nlimbs) {
  std::vector<uint64_t> tmp(nlimbs);
  seal::util::add_uint(op0, op1, nlimbs, tmp.data());
  if (seal::util::is_greater_than_or_equal_uint(tmp.data(), modulus, nlimbs)) {
    seal::util::sub_uint(tmp.data(), modulus, nlimbs, op0);
  } else {
    std::copy_n(tmp.data(), nlimbs, op0);
  }
}

void negate_uint_mod_inplace(uint64_t* op0, const uint64_t* modulus,
                             size_t nlimbs) {
  if (std::all_of(op0, op0 + nlimbs, [](uint64_t x) { return x == 0; })) {
    return;
  }

  std::vector<uint64_t> tmp(nlimbs);
  seal::util::sub_uint(modulus, op0, nlimbs, tmp.data());
  std::copy_n(tmp.data(), nlimbs, op0);
}

void sub_uint_mod_inplace(uint64_t* op0, const uint64_t* op1,
                          const uint64_t* modulus, size_t nlimbs) {
  std::vector<uint64_t> tmp(nlimbs);
  std::copy_n(op1, nlimbs, tmp.data());
  negate_uint_mod_inplace(tmp.data(), modulus, nlimbs);
  add_uint_mod_inplace(op0, tmp.data(), modulus, nlimbs);
}

template <typename T>
void TruncateThenReduce(const seal::Plaintext& pt0, const seal::Plaintext& pt1,
                        const seal::SEALContext::ContextData& cntxt,
                        size_t shift_amount, size_t out_width,
                        absl::Span<T> out0, absl::Span<T> out1) {
  SPU_ENFORCE(out_width > 0 and out_width <= 128,
              "Support out_width <= 64 for now");
  const auto& params = cntxt.parms();
  const auto& modulus = params.coeff_modulus();
  size_t N = params.poly_modulus_degree();
  size_t L = modulus.size();
  SPU_ENFORCE_EQ(pt0.coeff_count(), N * L);
  SPU_ENFORCE_EQ(pt1.coeff_count(), N * L);

  std::vector<uint64_t> limbs0(pt0.coeff_count());
  std::vector<uint64_t> limbs1(pt1.coeff_count());
  std::copy_n(pt0.data(), limbs0.size(), limbs0.data());
  std::copy_n(pt1.data(), limbs1.size(), limbs1.data());

  // From RNS format to bigInt format
  // TODO(lwj) optimize the cases L <= 2
  cntxt.rns_tool()->base_q()->compose_array(limbs0.data(), N,
                                            seal::MemoryManager::GetPool());
  cntxt.rns_tool()->base_q()->compose_array(limbs1.data(), N,
                                            seal::MemoryManager::GetPool());

  T mask = static_cast<T>(-1);
  if (out_width < sizeof(T) * 8) {
    mask = (static_cast<T>(1) << out_width) - 1;
  }

  std::vector<uint64_t> _tmp0(L);
  std::vector<uint64_t> _tmp1(L);
  auto tmp0 = _tmp0.data();
  auto tmp1 = _tmp1.data();

  using namespace seal::util;

  auto Q = cntxt.total_coeff_modulus();
  auto Qhalf = cntxt.upper_half_threshold();
  std::vector<uint64_t> bigval(L);
  right_shift_uint(Q, 2, L, bigval.data());

  std::vector<uint64_t> Qhalf_shifted(L);
  std::vector<uint64_t> Q_shifted(L);
  right_shift_uint(Q, shift_amount, L, Q_shifted.data());
  right_shift_uint(Qhalf, shift_amount, L, Qhalf_shifted.data());

  for (size_t i = 0; i < N; ++i) {
    auto shr0 = limbs0.data() + i * L;
    auto shr1 = limbs1.data() + i * L;
    // + Q/2 convert arith-shift to logical shift
    add_uint_mod_inplace(shr0, Qhalf, Q, L);

    right_shift_uint(shr0, shift_amount, L, tmp0);
    out0[i] = (tmp0[0] - Qhalf_shifted[0]) & mask;

    right_shift_uint(shr1, shift_amount, L, tmp1);
    out1[i] = tmp1[0] & mask;

    // TODO: apply the heuristic
    add_uint(shr0, shr1, L, tmp0);
    bool wrap = is_greater_than_or_equal_uint(tmp0, Q, L);
    out0[i] -= wrap * (Q_shifted[0] & mask);

    out0[i] &= mask;
  }
}

// y0 = round(Q/2^k*x0) mod Q
// y1 = round(Q/2^k*x1) mod Q
//
// x0 + x1 = x + w*2^k
//
//   Q/2^k*x0  + Q/2^k*x1 mod Q
// = Q/2^k*(x0 + x1) mod Q
// = Q/2^k*(x + w^k) mod Q
// = Q/2^k*x
// x0 + x1 in Zk
// y0 + y1 in Zq
template <typename T>
void ModulusExtend(absl::Span<const T> shr0, absl::Span<const T> shr1,
                   size_t input_width, int extra_delta,
                   const seal::SEALContext& context, seal::parms_id_type pid,
                   seal::Plaintext& out0, seal::Plaintext& out1) {
  SPU_ENFORCE(input_width > 0 and sizeof(T) * 8 >= input_width);
  SPU_ENFORCE(extra_delta > 0);

  size_t n = shr0.size();
  SPU_ENFORCE_EQ(n, shr1.size());

  auto cntxt = context.get_context_data(pid);
  const auto& params = cntxt->parms();
  const auto& modulus = params.coeff_modulus();
  size_t N = params.poly_modulus_degree();
  size_t L = modulus.size();
  SPU_ENFORCE(n <= N);
  out0.parms_id() = seal::parms_id_zero;
  out1.parms_id() = seal::parms_id_zero;
  out0.resize(N * L);
  out1.resize(N * L);

  EnableCPRNG cprng;
  cprng.UniformPoly(context, &out0, pid);
  using sT = typename std::make_signed<T>::type;
  for (size_t j = 0; j < L; ++j) {
    auto dst0 = out0.data() + j * n;
    auto dst1 = out1.data() + j * n;
    for (size_t i = 0; i < n; ++i) {
      sT x = shr0[i] + shr1[i];
      T ux = std::abs(x);
      ux <<= extra_delta;

      uint64_t v = BarrettReduce(ux, modulus[j]);
      if (std::signbit(x)) {
        v = seal::util::negate_uint_mod(v, modulus[j]);
      }
      dst1[i] = seal::util::sub_uint_mod(v, dst0[i], modulus[j]);
    }
  }

  out0.parms_id() = pid;
  out1.parms_id() = pid;
}

template <typename T>
void ModulusExtend(absl::Span<const T> shr0, absl::Span<const T> shr1,
                   size_t input_width, int extra_delta,
                   const seal::SEALContext::ContextData& cntxt,
                   seal::Plaintext& out0, seal::Plaintext& out1) {
  SPU_ENFORCE(input_width > 0 and sizeof(T) * 8 >= input_width);
  SPU_ENFORCE(extra_delta > 0);

  size_t n = shr0.size();
  SPU_ENFORCE_EQ(n, shr1.size());

  const auto& params = cntxt.parms();
  const auto& modulus = params.coeff_modulus();
  size_t N = params.poly_modulus_degree();
  size_t L = modulus.size();
  SPU_ENFORCE(n <= N);
  out0.parms_id() = seal::parms_id_zero;
  out1.parms_id() = seal::parms_id_zero;
  out0.resize(N * L);
  out1.resize(N * L);

  T upper = 0;
  if (input_width < sizeof(T) * 8) {
    upper = static_cast<T>(1) << input_width;
  }
  T half = static_cast<T>(1) << (input_width - 1);
  T mask = upper - 1;

  for (size_t j = 0; j < L; ++j) {
    using namespace seal::util;
    auto _scale = barrett_reduce_64(1UL << extra_delta, modulus[j]);
    MultiplyUIntModOperand scale;
    scale.set(_scale, modulus[j]);

    auto half_mod_q = barrett_reduce_64(half, modulus[j]);
    auto upper_mod_q = add_uint_mod(half_mod_q, half_mod_q, modulus[j]);

    for (size_t i = 0; i < n; ++i) {
      // Convert SignedExtension to ZeroExtension
      //
      // SExtend(x) = ZeroExtend(x + 2^{k - 1}) - 2^{k - 1} mod Q
      //
      // The following local computations are perform on share0.
      T two_component = (shr0[i] + half) & mask;

      // x0 + x1 >= 2^k
      // x0 >= -x1
      // TODO Use heuristic to compute the wrap
      bool wrap = (two_component >= (upper - shr1[i]));

      // Compute [x] - <wrap> * 2^k mod Q
      // The term 2^k mod Q (i.e., upper_mod_q)
      uint64_t t0 = barrett_reduce_64(two_component, modulus[j]);
      if (wrap) {
        t0 = sub_uint_mod(t0, upper_mod_q, modulus[j]);
      }
      // subtract 2^{k - 1} mod Q to convert to SignedExtension.
      // NOTE(lwj) lazy reduction here
      t0 = modulus[j].value() + t0 - half_mod_q;

      out0[j * N + i] = multiply_uint_mod(t0, scale, modulus[j]);
      out1[j * N + i] = multiply_uint_mod(shr1[i] & mask, scale, modulus[j]);
    }
  }

  out0.parms_id() = cntxt.parms_id();
  out1.parms_id() = cntxt.parms_id();
}

template <typename T>
void TruncateThenReduce(const seal::Plaintext& ckks_pt,
                        const seal::SEALContext::ContextData& cntxt,
                        size_t shift_amount, size_t out_width, bool is_rank0,
                        absl::Span<T> out) {
  SPU_ENFORCE(out_width > 0 and out_width <= 64,
              "Support out_width <= 64 for now");
  const auto& params = cntxt.parms();
  const auto& modulus = params.coeff_modulus();
  size_t N = params.poly_modulus_degree();
  size_t L = modulus.size();
  SPU_ENFORCE_EQ(ckks_pt.coeff_count(), N * L);

  std::vector<uint64_t> limbs(ckks_pt.coeff_count());
  std::copy_n(ckks_pt.data(), limbs.size(), limbs.data());

  // From RNS format to bigInt format
  // TODO(lwj) optimize the cases L <= 2
  cntxt.rns_tool()->base_q()->compose_array(limbs.data(), N,
                                            seal::MemoryManager::GetPool());

  T mask = static_cast<T>(-1);
  if (out_width < sizeof(T) * 8) {
    mask = (static_cast<T>(1) << out_width) - 1;
  }

  std::vector<uint64_t> _tmp(L);
  auto tmp = _tmp.data();

  if (is_rank0) {
    for (size_t i = 0; i < N; ++i) {
      using namespace seal::util;
      auto shr = limbs.data() + i * L;

      right_shift_uint(shr, shift_amount, L, tmp);
      out[i] = tmp[0] & mask;
    }
    return;
  }

  // rank1 uses a different local truncation
  auto Q = cntxt.total_coeff_modulus();

  std::vector<uint64_t> Qhalf(L);
  seal::util::right_shift_uint(Q, 1, L, Qhalf.data());
  // Q/2
  std::vector<uint64_t> Q_shifted(L);
  seal::util::right_shift_uint(Q, shift_amount, L, Q_shifted.data());

  for (size_t i = 0; i < N; ++i) {
    using namespace seal::util;
    auto shr = limbs.data() + i * L;

    right_shift_uint(shr, shift_amount, L, tmp);

    out[i] = (tmp[0] - Q_shifted[0]) & mask;
  }
}

// view [0, 2^k) as [-2^k/2, 2^k/2)
template <typename U>
auto ToSignType(U x, size_t width) {
  using S = typename std::make_signed<U>::type;
  if (sizeof(U) * 8 == width) {
    return static_cast<S>(x);
  }

  U half = static_cast<U>(1) << (width - 1);
  if (x >= half) {
    U upper = static_cast<U>(1) << width;
    x -= upper;
  }
  return static_cast<S>(x);
}

int main_back() {
  const size_t poly_N = 8192;
  const size_t nslots = poly_N / 2;
  const int scale = 30;
  const int mpc_fxp = 12;
  const int fft_fxp = std::log2(poly_N) + 12;  // mpc_fxp;

  using mpc_t = uint32_t;
  using signed_t = std::make_signed<mpc_t>::type;
  size_t mpc_width = 32;

  MPCCKKSEncoder<mpc_t> mpc_ckks_decoder(fft_fxp, poly_N);

  std::vector<int> modulus_bits = {60, 30, 50};
  auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

  EnableCPRNG cprng;

  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_use_special_prime(true);
  parms.set_poly_modulus_degree(poly_N);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext context(parms, true, seal::sec_level_type::none);

  seal::Evaluator evaluator(context);
  seal::KeyGenerator keygen(context);
  auto sk = keygen.secret_key();
  seal::PublicKey pk;
  keygen.create_public_key(pk);
  seal::RelinKeys rk;
  keygen.create_relin_keys(rk);

  seal::Encryptor encryptor(context, sk);
  seal::Decryptor decryptor(context, sk);
  seal::CKKSEncoder ckks_encoder(context);

  std::vector<double> slots(nslots);
  std::vector<double> coeff(poly_N);
  std::uniform_real_distribution<double> uniform(-8.0, 8.0);
  std::default_random_engine rdv(std::time(0));
  std::generate_n(slots.data(), slots.size(), [&]() { return uniform(rdv); });
  std::generate_n(coeff.data(), coeff.size(), [&]() { return uniform(rdv); });

  seal::Plaintext pt;
  ToCKKSPt(coeff, std::pow(2.0, scale), context, pt);
  NttInplace(pt, context);

  seal::Ciphertext ct;
  encryptor.encrypt_symmetric(pt, ct);

  seal::Plaintext rnd;
  cprng.UniformPoly(context, &rnd, ct.parms_id());

  InvNttInplace(ct, context);
  SubPlainInplace(ct, rnd, context);

  NttInplace(ct, context);
  seal::Plaintext decrypt;
  decryptor.decrypt(ct, decrypt);
  InvNttInplace(decrypt, context);

  std::vector<mpc_t> mpc0(poly_N);
  std::vector<mpc_t> mpc1(poly_N);

  TruncateThenReduce<mpc_t>(
      rnd, decrypt, *context.get_context_data(ct.parms_id()), scale - mpc_fxp,
      mpc_width, absl::MakeSpan(mpc0), absl::MakeSpan(mpc1));

  double trunc_err = 0.0;
  for (size_t i = 0; i < nslots; ++i) {
    signed_t a = ToSignType(mpc0[i] + mpc1[i], mpc_width);

    auto expected = coeff[i];
    double got = static_cast<double>(a) / (1LL << mpc_fxp);
    if (i < 8) {
      printf("%f => %f\n", expected, got);
    }
    trunc_err = std::max(trunc_err, std::abs(expected - got));
  }
  printf("max trunc error %f\n\n", trunc_err);

  ckks_encoder.encode(slots, std::pow(2.0, scale), pt);

  encryptor.encrypt_symmetric(pt, ct);

  // CKKS Computation Begin
  evaluator.square_inplace(ct);
  evaluator.relinearize_inplace(ct, rk);
  evaluator.rescale_to_next_inplace(ct);
  // CKKS Computation End
  cprng.UniformPoly(context, &rnd, ct.parms_id());

  InvNttInplace(ct, context);
  SubPlainInplace(ct, rnd, context);

  NttInplace(ct, context);
  decryptor.decrypt(ct, decrypt);
  InvNttInplace(decrypt, context);

  TruncateThenReduce<mpc_t>(
      rnd, decrypt, *context.get_context_data(ct.parms_id()), scale - mpc_fxp,
      mpc_width, absl::MakeSpan(mpc0), absl::MakeSpan(mpc1));

  // TruncateThenReduce<mpc_t>(rnd, *context.get_context_data(ct.parms_id()),
  //                           scale - mpc_fxp, mpc_width, false,
  //                           absl::MakeSpan(mpc0));
  // TruncateThenReduce<mpc_t>(decrypt,
  // *context.get_context_data(ct.parms_id()),
  //                           scale - mpc_fxp, mpc_width, true,
  //                           absl::MakeSpan(mpc1));

  std::vector<mpc_t> mpc_decode0(nslots);
  std::vector<mpc_t> mpc_decode1(nslots);

  // TODO(lwj): apply the O(n^2) algorithm to reduce the multiplication depth
  mpc_ckks_decoder.decode(absl::MakeSpan(mpc0), absl::MakeSpan(mpc_decode0));
  mpc_ckks_decoder.decode(absl::MakeSpan(mpc1), absl::MakeSpan(mpc_decode1));

  double simd_err = 0.0;
  for (size_t i = 0; i < nslots; ++i) {
    signed_t a = ToSignType(mpc_decode0[i] + mpc_decode1[i], mpc_width);

    auto expected = std::pow(slots[i], 2.0);
    auto got = a / std::pow(2., mpc_fxp);
    auto diff = std::abs(expected - got);
    if (i < 8) {
      printf("%f => %f\n", expected, got);
    }

    simd_err = std::max(simd_err, diff);
  }

  printf("max simd error %f\n", simd_err);
  return 0;
}

void multiply_scalar_inplace(seal::Ciphertext& ct, double scalar, double scale,
                             const seal::SEALContext& context) {
  int new_scale = std::log2(ct.scale() * scale);
  auto cntxt = context.get_context_data(ct.parms_id());

  SPU_ENFORCE(cntxt != nullptr);
  SPU_ENFORCE(new_scale + 1 < cntxt->total_coeff_modulus_bit_count());
  SPU_ENFORCE(std::log2(std::abs(scalar * scale)) <= 60);

  const auto& parms = cntxt->parms();
  const auto& modulus = parms.coeff_modulus();
  size_t n = parms.poly_modulus_degree();

  for (size_t k = 0; k < ct.size(); ++k) {
    uint64_t* dst_ptr = ct.data(k);
    for (size_t j = 0; j < modulus.size(); ++j) {
      uint64_t v = std::round(scale * std::abs(scalar));
      v = seal::util::barrett_reduce_64(v, modulus[j]);
      if (std::signbit(scalar)) {
        v = seal::util::negate_uint_mod(v, modulus[j]);
      }

      seal::util::multiply_poly_scalar_coeffmod(dst_ptr, n, v, modulus[j],
                                                dst_ptr);
      dst_ptr += n;
    }
  }

  ct.scale() = ct.scale() * scale;
}

// a2*x^2 + a3*x^3 + a4*x^4
void evaluate_poly4(const seal::Ciphertext& ct,
                    const seal::SEALContext& context, const seal::RelinKeys& rk,
                    absl::Span<const double> coeff3, seal::Ciphertext& out) {
  SPU_ENFORCE_EQ(coeff3.size(), 3U);
  auto B = context.get_context_data(ct.parms_id())
               ->next_context_data()
               ->total_coeff_modulus_bit_count();
  [[maybe_unused]] const double coeff_scale =
      std::pow(2., B - 2) / std::pow(ct.scale(), 2);

  seal::Evaluator eval(context);

  // x^2 scale= 2*fxp
  std::vector<seal::Ciphertext> powers(3);
  eval.square(ct, powers[0]);
  eval.relinearize_inplace(powers[0], rk);

  // x^3 scale = fxp
  eval.multiply(ct, powers[0], powers[1]);
  eval.relinearize_inplace(powers[1], rk);
  eval.rescale_to_next_inplace(powers[1]);

  // x^4 scale= 2*fxp
  auto cpy = ct;
  eval.mod_switch_to_next(ct, cpy);
  eval.multiply(cpy, powers[1], powers[2]);
  eval.relinearize_inplace(powers[2], rk);

  multiply_scalar_inplace(powers[0], coeff3[0], coeff_scale, context);
  multiply_scalar_inplace(powers[1], coeff3[1],
                          powers[0].scale() / powers[1].scale(), context);
  multiply_scalar_inplace(powers[2], coeff3[2], coeff_scale, context);

  eval.mod_switch_to_next(powers[0], out);
  powers[1].scale() = out.scale();
  powers[2].scale() = out.scale();
  eval.add_inplace(out, powers[1]);
  eval.add_inplace(out, powers[2]);
}

void test_modulus_extend() {
  const size_t poly_N = 8192;
  const size_t nslots = poly_N / 2;

  const int scale = 28;
  const int mpc_fxp = 18;
  std::vector<int> modulus_bits = {40, 40, 2 * scale, 60};

  const int fft_fxp = std::log2(poly_N) + 12;

  auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_use_special_prime(true);
  parms.set_poly_modulus_degree(poly_N);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext context(parms, true, seal::sec_level_type::none);

  using mpc_t = uint64_t;
  [[maybe_unused]] size_t mpc_width = 8 * sizeof(mpc_t);

  // real || imag parts
  std::vector<std::complex<double>> slots(nslots);
  std::uniform_real_distribution<double> uniform(-8.0, 8.0);
  std::default_random_engine rdv(std::time(0));

  std::uniform_int_distribution<mpc_t> mpc_uniform(0, static_cast<mpc_t>(-1));
  std::vector<std::complex<mpc_t>> mpc_vec0(nslots);
  std::vector<std::complex<mpc_t>> mpc_vec1(nslots);

  int rep = 10;
  for (int iter = 0; iter < rep; ++iter) {
    std::generate_n(reinterpret_cast<double*>(slots.data()), 2 * nslots,
                    [&]() { return uniform(rdv); });

    for (size_t i = 0; i < nslots; ++i) {
      mpc_vec0[i].real(mpc_uniform(rdv));
      mpc_vec0[i].imag(mpc_uniform(rdv));

      mpc_vec1[i].real(EncodeToFxp<mpc_t>(slots[i].real(), mpc_fxp) -
                       mpc_vec0[i].real());
      mpc_vec1[i].imag(EncodeToFxp<mpc_t>(slots[i].imag(), mpc_fxp) -
                       mpc_vec0[i].imag());
    }

    seal::CKKSEncoder ckks_encoder(context);
    seal::Plaintext pt;
    ckks_encoder.encode(slots, std::pow(2.0, scale), pt);

    std::vector<double> expected_coeffs(poly_N);
    InvNttInplace(pt, context);
    FromCKKSPt(pt, std::pow(2.0, scale), context,
               absl::MakeSpan(expected_coeffs));

    MPCCKKSEncoder<mpc_t> mpc_ckks_encoder(fft_fxp, poly_N);

    std::vector<mpc_t> mpc_encoded_vec0(poly_N);
    std::vector<mpc_t> mpc_encoded_vec1(poly_N);
    mpc_ckks_encoder.encode_complex(mpc_vec0, absl::MakeSpan(mpc_encoded_vec0));
    mpc_ckks_encoder.encode_complex(mpc_vec1, absl::MakeSpan(mpc_encoded_vec1));

    double encode_err = 0.0;
    for (size_t i = 0; i < poly_N; ++i) {
      double diff = std::abs(
          expected_coeffs[i] -
          ToSignType(mpc_encoded_vec0[i] + mpc_encoded_vec1[i], mpc_width) /
              std::pow(2., mpc_fxp));
      encode_err = std::max(encode_err, diff);
    }
    printf("encode error %f\n", encode_err);

    seal::Plaintext poly0;
    seal::Plaintext poly1;

    ModulusExtend<mpc_t>(absl::MakeSpan(mpc_encoded_vec0),
                         absl::MakeSpan(mpc_encoded_vec1), mpc_width,
                         scale - mpc_fxp, context, context.first_parms_id(),
                         poly0, poly1);

    NttInplace(poly0, context);
    NttInplace(poly1, context);

    poly0.scale() = std::pow(2.0, scale);
    poly1.scale() = std::pow(2.0, scale);

    seal::Evaluator evaluator(context);
    seal::KeyGenerator keygen(context);
    auto sk = keygen.secret_key();
    seal::Encryptor encryptor(context, sk);
    seal::Decryptor decryptor(context, sk);

    seal::Ciphertext ct;
    encryptor.encrypt_symmetric(poly0, ct);
    evaluator.add_plain_inplace(ct, poly1);

    seal::Plaintext dec;
    decryptor.decrypt(ct, dec);
    std::vector<std::complex<double>> got_slots;
    ckks_encoder.decode(dec, got_slots);

    for (size_t i = 0; i < 8; ++i) {
      printf("%f => %f; %f => %f\n", slots[i].real(), got_slots[i].real(),
             slots[i].imag(), got_slots[i].imag());
    }
  }
}

// HETransformer's method
void test_ckks_to_mpc(int scale) {
  const size_t poly_N = 8192;
  const size_t nslots = poly_N / 2;

  // Pr(\sqrt{N} * Delta * 2/ q0)
  std::vector<int> modulus_bits = {50, 40, 20};

  auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_use_special_prime(true);
  parms.set_poly_modulus_degree(poly_N);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext context(parms, true, seal::sec_level_type::none);

  seal::Evaluator evaluator(context);
  seal::KeyGenerator keygen(context);
  seal::PublicKey pk;
  auto sk = keygen.secret_key();
  keygen.create_public_key(pk);

  seal::Encryptor encryptor(context, pk);
  seal::Decryptor decryptor(context, sk);
  seal::CKKSEncoder ckks_encoder(context);

  int64_t q0 = modulus[0].value();
  int64_t q0_half = q0 >> 1;
  std::uniform_int_distribution<uint64_t> uint_uniform(
      0, static_cast<uint64_t>(q0) - 1);

  auto round_to = [&](double x, int s) {
    uint64_t u = std::floor(std::abs(x) * (1L << s));
    u = seal::util::barrett_reduce_64(u, modulus[0]);
    if (std::signbit(x)) {
      u = seal::util::negate_uint_mod(u, modulus[0]);
    }
    return u;
  };

  int count_err = 0;
  size_t ntrial = (1L << 26) / poly_N;

  for (size_t rep = 0; rep < ntrial; ++rep) {
    // Step 0: Prepare an CKKS ciphertext
    std::vector<std::complex<double>> real_slots(nslots);

    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::normal_distribution gd(16.0, 8.0);

    std::default_random_engine rdv(std::time(0));
    std::generate_n(reinterpret_cast<double*>(real_slots.data()), 2 * nslots,
                    [&]() { return uniform(rdv); });

    // std::generate_n(reinterpret_cast<double*>(real_slots.data()), 2 * nslots,
    //                 []() { return 10.0; });

    seal::Plaintext pt;
    seal::Ciphertext ct;
    ckks_encoder.encode(real_slots, static_cast<double>(1L << scale), pt);
    encryptor.encrypt(pt, ct);

    // Perform HE-to-MPC
    // Step 1: Down to mod q0
    evaluator.mod_switch_to_inplace(ct, context.last_parms_id());

#if 1
    // Step 2: Sample a random poly `r` from Rq0
    seal::Plaintext random_poly;
    random_poly.resize(poly_N);
    EnableCPRNG cprng;
    cprng.UniformPoly(context, &random_poly, context.last_parms_id());
    random_poly.parms_id() = context.last_parms_id();

    // Step 3: Decode the `r` regarding its scalaing factor as Delta
    random_poly.scale() = static_cast<double>(1L << scale);
    std::vector<uint64_t> share0(poly_N);

    // Step 4: round the decoded vector to mod q0 integers
    {
      std::vector<std::complex<double>> decoding_vec(nslots);
      ckks_encoder.decode(random_poly, decoding_vec);

      for (size_t i = 0; i < nslots; ++i) {
        share0[i] = round_to(-decoding_vec[i].real(), scale);
        share0[nslots + i] = round_to(-decoding_vec[i].imag(), scale);
      }
    }
#else
    // sample from (ZZ_q0)^N then encode it to Rq0
    std::vector<uint64_t> share0(poly_N);
    seal::Plaintext random_poly;
    random_poly.resize(poly_N);
    EnableCPRNG cprng;
    cprng.UniformPoly(context, &random_poly, context.last_parms_id());
    std::copy_n(random_poly.data(), poly_N, share0.data());

    {
      std::vector<std::complex<double>> decoding_vec(nslots);

      double l2_norm = 0.0;
      for (size_t i = 0; i < nslots; ++i) {
        int64_t u = share0[i];
        if (u >= q0_half) {
          u -= q0;
        }
        l2_norm += static_cast<double>(u) * static_cast<double>(u);

        decoding_vec[i].real(u / static_cast<double>(1L << scale));
        // decoding_vec[i].real(static_cast<double>(u));

        u = share0[nslots + i];
        if (u >= q0_half) {
          u -= q0;
        }
        l2_norm += static_cast<double>(u) * static_cast<double>(u);

        decoding_vec[i].imag(u / static_cast<double>(1L << scale));
        // decoding_vec[i].imag(static_cast<double>(u));
      }
      l2_norm = std::sqrt(l2_norm);
      rnd_mean += l2_norm;

      ckks_encoder.encode(decoding_vec, static_cast<double>(1L << scale),
                          random_poly);
      // ckks_encoder.encode(decoding_vec, 1.0, random_poly);
      random_poly.scale() = 1L << scale;

      InvNttInplace(random_poly, context);

      for (size_t i = 0; i < poly_N; ++i) {
        int64_t u = random_poly[i];
        if (u >= q0_half) {
          u -= q0;
        }

        mean += static_cast<double>(u);
        max = std::max(max, static_cast<double>(u));
        min = std::min(min, static_cast<double>(u));
      }

      NttInplace(random_poly, context);
    }
#endif

    // Step 5. Do ciphertext masking
    evaluator.add_plain_inplace(ct, random_poly);

    // Step 6: decrypt the masked ct and perform decoding
    decryptor.decrypt(ct, pt);

    std::vector<uint64_t> share1(poly_N);
    {
      std::vector<std::complex<double>> decoding_vec(nslots);
      ckks_encoder.decode(pt, decoding_vec);

      for (size_t i = 0; i < nslots; ++i) {
        share1[i] = round_to(decoding_vec[i].real(), scale);
        share1[nslots + i] = round_to(decoding_vec[i].imag(), scale);
      }
    }

    // Check correctness
    // int64_t real_err = 0;
    // int64_t imag_err = 0;

    for (size_t i = 0; i < nslots; ++i) {
      int64_t m0 = seal::util::add_uint_mod(share0[i], share1[i], modulus[0]);
      int64_t m1 = seal::util::add_uint_mod(share0[nslots + i],
                                            share1[nslots + i], modulus[0]);

      if (m0 >= q0_half) {
        m0 -= q0;
      }
      if (m1 >= q0_half) {
        m1 -= q0;
      }

      int64_t expected = real_slots[i].real() * (1L << scale);
      int64_t got = m0;
      size_t err = std::abs(expected - got);

      if (err > 2 * poly_N) {
        count_err += 1;
      }

      expected = real_slots[i].imag() * (1L << scale);
      got = m1;
      err = std::abs(expected - got);
      if (err > 2 * poly_N) {
        count_err += 1;
      }
      // imag_err = std::max(imag_err, std::abs(expected - got));
    }
  }

  // sqrt(N) * Delta / (2*q)
  printf("expected error ratio 2^%f\n",
         std::log2(std::sqrt(poly_N)) + scale - modulus[0].bit_count());

  printf("scale %d, actual error ratio 2^%f #error = %d\n", scale,
         std::log2(count_err * 1.0 / (ntrial * poly_N)), count_err);
}

template <typename T>
void ConvToMPC(absl::Span<const double> real, absl::Span<T> shr0,
               absl::Span<T> shr1, int fxp) {
  std::default_random_engine rdv(std::time(0));
  std::uniform_int_distribution<T> mpc_uniform(0, static_cast<T>(-1));
  size_t nsize = real.size();
  for (size_t i = 0; i < nsize; ++i) {
    shr0[i] = mpc_uniform(rdv);
    shr1[i] = EncodeToFxp<T>(real[i], fxp) - shr0[i];
  }
}

/**
 * To run this demo, we need to modify the SEAL's codes
 *
 * https://github.com/microsoft/SEAL/blob/main/native/src/seal/ckks.cpp#L34
 * Change the `uint64_t gen = 3;` to `uint64_t gen = 5;`
 *
 * https://github.com/microsoft/SEAL/blob/main/native/src/seal/util/galois.h#L169
 * Change `generator_ = 3;` to `generator_ = 5;`
 *
 * */
int main() {
  const size_t poly_N = 8192;
  const size_t nslots = poly_N / 2;
  const double apprx_range = 4.0;

  const int mpc_fxp = 20;
  const int fft_fxp = std::log2(poly_N) + 12;
  const int scale = 28;

  using mpc_t = uint64_t;
  // NOTE(lwj): we need to peform the local CKKS encoding in a larger ring size.
  using dmpc_t = uint128_t;
  size_t mpc_width = 8 * sizeof(mpc_t);

  std::vector<int> modulus_bits = {50, 40, 2 * scale, 60};

  printf("Q = %dbits\n",
         std::accumulate(modulus_bits.begin(), modulus_bits.end() - 1, 0));

  auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

  MPCCKKSEncoder<dmpc_t> dmpc_ckks_decoder(fft_fxp, poly_N);
  MPCCKKSEncoder<mpc_t> mpc_ckks_decoder(fft_fxp, poly_N);

  EnableCPRNG cprng;

  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_use_special_prime(true);
  parms.set_poly_modulus_degree(poly_N);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext context(parms, true, seal::sec_level_type::none);

  seal::Evaluator evaluator(context);
  seal::KeyGenerator keygen(context);
  auto sk = keygen.secret_key();
  seal::PublicKey pk;
  keygen.create_public_key(pk);
  seal::RelinKeys rk;
  keygen.create_relin_keys(rk);
  seal::GaloisKeys gk;
  keygen.create_galois_keys(gk);

  seal::Encryptor encryptor(context, sk);
  seal::Decryptor decryptor(context, sk);
  seal::CKKSEncoder ckks_encoder(context);

  std::vector<double> real_slots(nslots);
  std::vector<double> imag_slots(nslots);

  std::vector<dmpc_t> real_mpc_vec0(nslots);
  std::vector<dmpc_t> real_mpc_vec1(nslots);
  std::vector<dmpc_t> imag_mpc_vec0(nslots);
  std::vector<dmpc_t> imag_mpc_vec1(nslots);
  std::vector<std::complex<dmpc_t>> compx_mpc_vec0(nslots);
  std::vector<std::complex<dmpc_t>> compx_mpc_vec1(nslots);
  std::vector<mpc_t> mpc_poly_coeff0(poly_N);
  std::vector<mpc_t> mpc_poly_coeff1(poly_N);
  seal::Plaintext poly0;
  seal::Plaintext poly1;

  std::vector<std::complex<double>> complex_slots(nslots);

  double real_err = 0.0;
  double imag_err = 0.0;
  int trial = (1L << 23) / poly_N;
  printf("trials %d, N = %zd, Delta = %d bit\n", trial, poly_N, scale);

  std::random_device rdv;
  for (int rep = 0; rep < trial; ++rep) {
    // sample reals from [0.0, 4.0)
    std::uniform_real_distribution<double> uniform(0.0, apprx_range);
    std::generate_n(real_slots.data(), nslots, [&]() { return uniform(rdv); });
    std::generate_n(imag_slots.data(), nslots, [&]() { return uniform(rdv); });

    // Prepare MPC of fixed-point vector
    ConvToMPC<dmpc_t>(absl::MakeSpan(real_slots), absl::MakeSpan(real_mpc_vec0),
                      absl::MakeSpan(real_mpc_vec1), mpc_fxp - 1);

    ConvToMPC<dmpc_t>(absl::MakeSpan(imag_slots), absl::MakeSpan(imag_mpc_vec0),
                      absl::MakeSpan(imag_mpc_vec1), mpc_fxp - 1);

    // Each party encodes his local share to a poly over the MPC modulus
    // NOTE(lwj): we do arith right-shift because the real/imag seperation will
    // introduce a factor of 2.
    for (size_t i = 0; i < nslots; ++i) {
      compx_mpc_vec0[i].real(real_mpc_vec0[i]);
      compx_mpc_vec0[i].imag(imag_mpc_vec0[i]);

      compx_mpc_vec1[i].real(real_mpc_vec1[i]);
      compx_mpc_vec1[i].imag(imag_mpc_vec1[i]);
    }

    std::vector<dmpc_t> dmpc_poly_coeff0(poly_N);
    std::vector<dmpc_t> dmpc_poly_coeff1(poly_N);
    // local CKKS encoding over the MPC modulus 2^k
    dmpc_ckks_decoder.encode_complex(absl::MakeConstSpan(compx_mpc_vec0),
                                     absl::MakeSpan(dmpc_poly_coeff0));
    dmpc_ckks_decoder.encode_complex(absl::MakeConstSpan(compx_mpc_vec1),
                                     absl::MakeSpan(dmpc_poly_coeff1));
    // ring change from u128 to u64
    std::transform(dmpc_poly_coeff0.begin(), dmpc_poly_coeff0.end(),
                   mpc_poly_coeff0.data(),
                   [](dmpc_t x) -> mpc_t { return static_cast<mpc_t>(x); });
    std::transform(dmpc_poly_coeff1.begin(), dmpc_poly_coeff1.end(),
                   mpc_poly_coeff1.data(),
                   [](dmpc_t x) -> mpc_t { return static_cast<mpc_t>(x); });

    // Simulate a MPC protocol to convert modulus from to modulus Q
    ModulusExtend<mpc_t>(absl::MakeSpan(mpc_poly_coeff0),
                         absl::MakeSpan(mpc_poly_coeff1), mpc_width,
                         scale - mpc_fxp, context, context.first_parms_id(),
                         poly0, poly1);

    NttInplace(poly0, context);
    NttInplace(poly1, context);

    poly0.scale() = 1L << scale;
    poly1.scale() = 1L << scale;

    // Alice sends to Bob his ct Enc(poly0)
    seal::Ciphertext ct;
    encryptor.encrypt_symmetric(poly0, ct);
    // Bob local reconstruct the ct Enc(poly0 + poly1 mod Q)
    // which should decrypt to the CKKS encoding of the target vector.
    evaluator.add_plain_inplace(ct, poly1);
    // Split real/imag parts via HE conjugation (needs one KeySwitch)
    seal::Ciphertext ct1;
    {
      seal::Ciphertext tmp;
      // x + y*j => x - y*j
      evaluator.complex_conjugate(ct, gk, tmp);
      // x + y*j - (x - y*j)
      // - 2 * y*j
      evaluator.sub(tmp, ct, ct1);
      // - 2 * y*j => 2*y
      MulImageUnitInplace(ct1, context);
      // x + y*j + (x - y*j) => 2*x
      evaluator.add_inplace(ct, tmp);
    }

    // Finally the CKKS computation on Bob's side
    // BOLT's degree-4 polys for GELU
    std::vector<double> coeff2 = {0.001620808531841547, -0.03798164612714154};
    std::vector<double> coeff3 = {0.5410550166368381, -0.18352506127082727,
                                  0.020848611754127593};
    // Evaluate a polynomial f(x) = a2*x^2 + a3*x^3 + a4*x^4
    seal::Ciphertext gelu_ct;
    seal::Ciphertext gelu_ct1;
    evaluate_poly4(ct, context, rk, coeff3, gelu_ct);
    evaluate_poly4(ct1, context, rk, coeff3, gelu_ct1);
    // Merge the two ciphertexts into one
    InvNttInplace(gelu_ct, context);
    InvNttInplace(gelu_ct1, context);
    MulImageUnitInplace(gelu_ct1, context);
    seal::Ciphertext final_ct;
    evaluator.add(gelu_ct, gelu_ct1, final_ct);
    // CKKS Computation End

    seal::Plaintext rnd;
    seal::Plaintext decrypt;
    // sample r from Rq
    cprng.UniformPoly(context, &rnd, final_ct.parms_id());

    InvNttInplace(final_ct, context);
    // mask the cipher: ct - r
    SubPlainInplace(final_ct, rnd, context);

    NttInplace(final_ct, context);

    decryptor.decrypt(final_ct, decrypt);
    InvNttInplace(decrypt, context);

    std::vector<mpc_t> mpc0(poly_N);
    std::vector<mpc_t> mpc1(poly_N);

    // Convert mod q (poly) to mod 2^k poly
    // And handle the ckks Delta to mpc's fixed point scaling.
    TruncateThenReduce<mpc_t>(
        rnd, decrypt, *context.get_context_data(final_ct.parms_id()),
        static_cast<int>(std::log2(final_ct.scale())) - mpc_fxp, mpc_width,
        absl::MakeSpan(mpc0), absl::MakeSpan(mpc1));

    std::vector<mpc_t> mpc_real0(nslots);
    std::vector<mpc_t> mpc_real1(nslots);

    std::vector<mpc_t> mpc_imag0(nslots);
    std::vector<mpc_t> mpc_imag1(nslots);

    // Alice local decoding
    mpc_ckks_decoder.decode_complex(absl::MakeSpan(mpc0),
                                    absl::MakeSpan(mpc_real0),
                                    absl::MakeSpan(mpc_imag0));
    // Bob local decoding
    mpc_ckks_decoder.decode_complex(absl::MakeSpan(mpc1),
                                    absl::MakeSpan(mpc_real1),
                                    absl::MakeSpan(mpc_imag1));
    // Correctness check
    [[maybe_unused]] auto gelu_func = [](double x) {
      return 0.5 * x *
             (1.0 + std::tanh(std::sqrt(2.0 / M_PI) *
                              (x + 0.044715 * std::pow(x, 3.0))));
    };

    for (size_t i = 0; i < nslots; ++i) {
      double got_real = ToSignType(mpc_real0[i] + mpc_real1[i], mpc_width) /
                        static_cast<double>(1L << mpc_fxp);
      double got_imag = ToSignType(mpc_imag0[i] + mpc_imag1[i], mpc_width) /
                        static_cast<double>(1L << mpc_fxp);

      auto expected_real = gelu_func(real_slots[i]) - 0.5 * real_slots[i] -
                           coeff2[0] - coeff2[1] * std::abs(real_slots[i]);
      auto expected_imag = gelu_func(imag_slots[i]) - 0.5 * imag_slots[i] -
                           coeff2[0] - coeff2[1] * std::abs(imag_slots[i]);

      auto diff = std::abs(expected_real - got_real);
      real_err = std::max(real_err, diff);

      diff = std::abs(expected_imag - got_imag);
      imag_err = std::max(imag_err, diff);
    }

    if (rep % 10 == 0) {
      printf("until %d: max simd error %f %f\n", rep, real_err, imag_err);
    }
  }

  printf("max simd error %f %f\n", real_err, imag_err);
  return 0;
}
