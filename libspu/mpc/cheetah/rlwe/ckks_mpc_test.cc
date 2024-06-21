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
  SPU_ENFORCE(out_width > 0 and out_width <= 64,
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

// x0 + x1 in Zk
// y0 + y1 in Zq
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
  const double coeff_scale = std::pow(2., B - 2) / std::pow(ct.scale(), 2);

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
  const size_t poly_N = 1024;
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
  [[maybe_unused]] size_t mpc_width = 64;

  std::vector<double> slots(nslots);
  std::uniform_real_distribution<double> uniform(-8.0, 8.0);
  std::default_random_engine rdv(std::time(0));
  std::generate_n(slots.data(), nslots, [&]() { return uniform(rdv); });

  std::uniform_int_distribution<mpc_t> mpc_uniform(0, static_cast<mpc_t>(-1));
  std::vector<mpc_t> mpc_vec0(nslots);
  std::vector<mpc_t> mpc_vec1(nslots);

  for (size_t i = 0; i < nslots; ++i) {
    mpc_vec0[i] = mpc_uniform(rdv);
    mpc_vec1[i] = EncodeToFxp<mpc_t>(slots[i], mpc_fxp) - mpc_vec0[i];
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
  mpc_ckks_encoder.encode(mpc_vec0, absl::MakeSpan(mpc_encoded_vec0));
  mpc_ckks_encoder.encode(mpc_vec1, absl::MakeSpan(mpc_encoded_vec1));

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

  ModulusExtend<mpc_t>(
      absl::MakeSpan(mpc_encoded_vec0), absl::MakeSpan(mpc_encoded_vec1),
      mpc_width, scale - mpc_fxp, *context.first_context_data(), poly0, poly1);

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
  std::vector<double> got_slots;
  ckks_encoder.decode(dec, got_slots);

  for (size_t i = 0; i < 8; ++i) {
    printf("%f => %f\n", slots[i], got_slots[i]);
  }
}

int main() {
  test_modulus_extend();
  return 0;
  const size_t poly_N = 8192;
  const size_t nslots = poly_N / 2;
  const double apprx_range = 4.0;

  const int scale = 28;
  const int mpc_fxp = 20;
  const int fft_fxp = std::log2(poly_N) + 12;

  using mpc_t = uint64_t;
  size_t mpc_width = 64;

  std::vector<int> modulus_bits = {40, 40, 2 * scale, 60};

  printf("Q = %dbits\n",
         std::accumulate(modulus_bits.begin(), modulus_bits.end() - 1, 0));

  auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

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
  std::vector<std::complex<double>> complex_slots(nslots);

  std::uniform_real_distribution<double> uniform(0.0, apprx_range);
  std::default_random_engine rdv(std::time(0));
  std::generate_n(real_slots.data(), nslots, [&]() { return uniform(rdv); });
  std::generate_n(imag_slots.data(), nslots, [&]() { return uniform(rdv); });

  for (size_t i = 0; i < nslots; ++i) {
    complex_slots[i].real(0.5 * real_slots[i]);
    complex_slots[i].imag(0.5 * imag_slots[i]);
  }

  seal::Plaintext pt;
  seal::Ciphertext ct;
  seal::Ciphertext ct1;

  ckks_encoder.encode(complex_slots, std::pow(2.0, scale), pt);
  encryptor.encrypt_symmetric(pt, ct);
  size_t sent_bytes = EncodeSEALObject(ct).size() >> 1;

  // x - y*j
  {
    seal::Ciphertext tmp;
    evaluator.complex_conjugate(ct, gk, tmp);

    // x + yj - (x - yj)
    // 2 * yj
    evaluator.sub(ct, tmp, ct1);
    evaluator.negate_inplace(ct1);
    MulImageUnitInplace(ct1, context);

    evaluator.add_inplace(ct, tmp);
  }

  // ckks_encoder.encode(imag_slots, std::pow(2.0, scale), pt);
  // encryptor.encrypt_symmetric(pt, ct1);

  // CKKS Computation Begin
  std::vector<double> coeff3 = {0.5410550166368381, -0.18352506127082727,
                                0.020848611754127593};
  seal::Ciphertext gelu_ct;
  seal::Ciphertext gelu_ct1;
  evaluate_poly4(ct, context, rk, coeff3, gelu_ct);
  evaluate_poly4(ct1, context, rk, coeff3, gelu_ct1);

  InvNttInplace(gelu_ct, context);
  InvNttInplace(gelu_ct1, context);
  MulImageUnitInplace(gelu_ct1, context);
  evaluator.add(gelu_ct, gelu_ct1, ct);
  // CKKS Computation End
  size_t recv_bytes = EncodeSEALObject(ct).size();

  printf("%zd gelu sent %zd B, recv %zd B, %f B per\n", nslots * 2, sent_bytes,
         recv_bytes, (sent_bytes + recv_bytes) * 1.0 / (2 * nslots));

  seal::Plaintext rnd;
  seal::Plaintext decrypt;
  cprng.UniformPoly(context, &rnd, ct.parms_id());

  InvNttInplace(ct, context);
  SubPlainInplace(ct, rnd, context);

  NttInplace(ct, context);

  decryptor.decrypt(ct, decrypt);
  InvNttInplace(decrypt, context);

  std::vector<mpc_t> mpc0(poly_N);
  std::vector<mpc_t> mpc1(poly_N);

  TruncateThenReduce<mpc_t>(
      rnd, decrypt, *context.get_context_data(ct.parms_id()),
      static_cast<int>(std::log2(ct.scale())) - mpc_fxp, mpc_width,
      absl::MakeSpan(mpc0), absl::MakeSpan(mpc1));

  // TruncateThenReduce<mpc_t>(rnd, *context.get_context_data(ct.parms_id()),
  //                           scale - mpc_fxp, mpc_width, false,
  //                           absl::MakeSpan(mpc0));
  // TruncateThenReduce<mpc_t>(decrypt,
  // *context.get_context_data(ct.parms_id()),
  //                           scale - mpc_fxp, mpc_width, true,
  //                           absl::MakeSpan(mpc1));

  std::vector<mpc_t> mpc_real0(nslots);
  std::vector<mpc_t> mpc_real1(nslots);

  std::vector<mpc_t> mpc_imag0(nslots);
  std::vector<mpc_t> mpc_imag1(nslots);

  // TODO(lwj): apply the O(n^2) algorithm to reduce the multiplication depth
  mpc_ckks_decoder.decode_complex(absl::MakeSpan(mpc0),
                                  absl::MakeSpan(mpc_real0),
                                  absl::MakeSpan(mpc_imag0));
  mpc_ckks_decoder.decode_complex(absl::MakeSpan(mpc1),
                                  absl::MakeSpan(mpc_real1),
                                  absl::MakeSpan(mpc_imag1));
  // 0.001620808531841547, -0.03798164612714154

  double real_err = 0.0;
  double imag_err = 0.0;

  auto gelu_func = [](double x) {
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
                         0.001620808531841547 -
                         (-0.03798164612714154) * real_slots[i];
    auto expected_imag = gelu_func(imag_slots[i]) - 0.5 * imag_slots[i] -
                         0.001620808531841547 -
                         (-0.03798164612714154) * imag_slots[i];

    auto diff = std::abs(expected_real - got_real);
    real_err = std::max(real_err, diff);

    diff = std::abs(expected_imag - got_imag);
    imag_err = std::max(imag_err, diff);

    if (i < 8) {
      printf("%f => %f\t%f => %f\n", expected_real, got_real, expected_imag,
             got_imag);
    }
  }

  printf("max simd error %f %f\n", real_err, imag_err);
  return 0;
}
