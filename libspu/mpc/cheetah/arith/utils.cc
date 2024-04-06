#include "libspu/mpc/cheetah/arith/utils.h"

#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

namespace {
template <typename T>
T makeMask(size_t bits) {
  if (sizeof(T) * 8 == bits) {
    return -1;
  }
  return (static_cast<T>(1) << bits) - 1;
}
}  // namespace

namespace spu::mpc::cheetah {

RingShareToPrimeShareLocalConvertor::RingShareToPrimeShareLocalConvertor(
    size_t ring_width, const seal::Modulus& target_modulus)
    : ring_width_(ring_width), target_modulus_(target_modulus) {
  SPU_ENFORCE(ring_width > 0 and ring_width <= 128, "invalid ring width={}",
              ring_width);
  // limbs for 2^k
  if (static_cast<int>(ring_width) < target_modulus.bit_count()) {
    offset_ = 1ULL << ring_width;
  } else {
    uint64_t _2k[3] = {0, 0, 0};
    _2k[ring_width >> 6] = 1UL << (ring_width % 64);
    offset_ = seal::util::modulo_uint(_2k, 3, target_modulus_);
  }
  offset_ = seal::util::negate_uint_mod(offset_, target_modulus_);
}

NdArrayRef RingShareToPrimeShareLocalConvertor::Compute(
    const NdArrayRef& ring_share, int rank) const {
  SPU_ENFORCE(rank >= 0 and rank <= 1, "invalid rank={}", rank);
  auto elt = ring_share.eltype();
  SPU_ENFORCE(elt.isa<RingTy>());
  auto ft = elt.as<RingTy>()->field();
  SPU_ENFORCE(SizeOf(ft) * 8 >= ring_width_, "field type {} out-of-bound", ft);

  NdArrayRef out(elt, ring_share.shape());
  DISPATCH_ALL_FIELDS(ft, "r2f_cast", [&]() {
    NdArrayView<const ring2k_t> inp(ring_share);
    NdArrayView<ring2k_t> oup(out);
    auto msk = makeMask<ring2k_t>(ring_width_);
    if (static_cast<int>(ring_width_) < target_modulus_.bit_count()) {
      pforeach(0, ring_share.numel(),
               [&](int64_t i) { oup[i] = inp[i] & msk; });
    } else {
      pforeach(0, ring_share.numel(), [&](int64_t i) {
        oup[i] = BarrettReduce(inp[i] & msk, target_modulus_);
      });
    }

    if (rank != 0) {
      // x1 mod p
      return;
    }

    // x0 + (-2^k mod p)
    pforeach(0, ring_share.numel(), [&](int64_t i) {
      oup[i] = seal::util::add_uint_mod(static_cast<uint64_t>(oup[i]), offset_,
                                        target_modulus_);
    });
  });

  return out;
}

}  // namespace spu::mpc::cheetah
