#include <numeric>
#include <sstream>

#include "seal/seal.h"
#include "utils.h"

#include "libspu/mpc/cheetah/rlwe/utils.h"

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

void evaluate_to_power4(const seal::Ciphertext& ct,
                        const seal::SEALContext& context,
                        const seal::RelinKeys& rk,
                        std::vector<seal::Ciphertext>& powers) {
  powers.resize(3, seal::Ciphertext());
  seal::Evaluator eval(context);
  // Delta = 30bit
  // Q = {59, 60}
  //
  // x^2 scale= 30 + 30 = 60
  eval.square(ct, powers[0]);
  eval.relinearize_inplace(powers[0], rk);

  // x^3 scale = 60 + 30 - 60 = 30
  eval.multiply(ct, powers[0], powers[1]);
  eval.relinearize_inplace(powers[1], rk);
  eval.rescale_to_next_inplace(powers[1]);

  // x^4 scale= 30 + 30 = 60
  auto cpy = ct;
  eval.mod_switch_to_next(ct, cpy);
  eval.multiply(cpy, powers[1], powers[2]);
  eval.relinearize_inplace(powers[2], rk);

  auto pid = context.last_parms_id();
  for (auto& ct : powers) {
    eval.mod_switch_to_inplace(ct, pid);
  }
}

int main() {
  size_t poly_N = 8192;
  std::vector<int> modulus_bits;
  double apprx_range = 4.0;
  int apprx_nbits = std::ceil(std::log2(std::pow(apprx_range, 4.0)));
  int delta_nbits = std::min<int>((59 - apprx_nbits) / 2.0, 12 + 13);

  modulus_bits = {delta_nbits * 2 + apprx_nbits + 1, delta_nbits * 2,
                  delta_nbits * 2};
  printf("approximate [%f, %f] Delta %d bits\n", -apprx_range, apprx_range,
         delta_nbits);
  printf("Q = %dbits\n",
         std::accumulate(modulus_bits.begin(), modulus_bits.end() - 1, 0));

  auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_use_special_prime(true);
  parms.set_poly_modulus_degree(poly_N);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext context(parms, true, seal::sec_level_type::tc128);

  seal::Evaluator evaluator(context);
  seal::KeyGenerator keygen(context);
  auto sk = keygen.secret_key();
  seal::RelinKeys rk;
  keygen.create_relin_keys(rk);
  seal::GaloisKeys gk;
  keygen.create_galois_keys(gk);

  seal::Encryptor encryptor(context, sk);
  seal::Decryptor decryptor(context, sk);
  seal::CKKSEncoder encoder(context);

  // a + b*j => 2a, 2b*j
  //
  // (a + b*j) * (a + b*j)
  // (a^2 - b^2) + 2a*b*j
  size_t nslots = poly_N / 2;
  std::vector<double> input(nslots);
  std::uniform_real_distribution<double> uniform(0.0, apprx_range);
  std::default_random_engine rdv;
  std::generate_n(input.begin(), input.size(), [&]() { return uniform(rdv); });

  double delta = std::pow(2.0, delta_nbits);
  seal::Plaintext pt;
  encoder.encode(input, delta, pt);
  seal::Ciphertext ct;

  encryptor.encrypt_symmetric(pt, ct);
  size_t sent_bytes = spu::mpc::cheetah::EncodeSEALObject(ct).size();
  printf("sent |ct| %f KB\n", sent_bytes / 2.0 / 1024.0);

  std::vector<seal::Ciphertext> pow4;
  evaluate_to_power4(ct, context, rk, pow4);

  // 1 ct of N values
  // a + b*j => a, b
  //
  // a => a^2, a^3, a^4
  // b => b^2, b^3, b^4
  //
  // 3 ct of N values
  // a^2 + b^2*j
  // a^3 + b^4*j
  // a^4 + b^4*j
  MulImageUnitInplace(pow4[2], context);
  pow4[0].scale() = pow4[2].scale();
  evaluator.add_inplace(pow4[0], pow4[2]);
  // pow4.pop_back();

  size_t response_bytes = 0;
  for (auto& ct : pow4) {
    response_bytes += spu::mpc::cheetah::EncodeSEALObject(ct).size();
  }
  printf("response |ct| %f KB\n", response_bytes / 1024.0);

  std::vector<std::complex<double>> pow_2_4_slots;
  std::vector<double> pow_3_slots;

  {
    seal::Plaintext pt;
    decryptor.decrypt(pow4[0], pt);
    encoder.decode(pt, pow_2_4_slots);

    decryptor.decrypt(pow4[1], pt);
    encoder.decode(pt, pow_3_slots);
  }

  {
    double err = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
      double expected = std::pow(input[i], 2);
      double got = pow_2_4_slots[i].real();
      double e = std::abs(expected - got);
      err = std::max(err, e);
    }
    printf("pow2 %f\n", err);

    err = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
      double expected = std::pow(input[i], 4);
      double got = ((i % 2 == 0) ? 1.0 : -1.0) * pow_2_4_slots[i].imag();
      double e = std::abs(expected - got);
      err = std::max(err, e);
    }
    printf("pow4 %f\n", err);

    err = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
      double expected = std::pow(input[i], 3);
      double got = pow_3_slots[i];
      double e = std::abs(expected - got);
      err = std::max(err, e);
    }
    printf("pow3 %f\n", err);
  }

  printf("%f per B\n", (sent_bytes + response_bytes) * 1.0 / (2 * nslots));

  return 0;
}
