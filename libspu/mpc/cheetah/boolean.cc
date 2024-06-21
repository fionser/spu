// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/cheetah/boolean.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"

namespace spu::mpc::cheetah {

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());

  int64_t numel = lhs.numel();
  if (numel == 0) {
    return NdArrayRef(lhs.eltype(), lhs.shape());
  }

  return TiledDispatchOTFunc(
             ctx, lhs, rhs,
             [&](const NdArrayRef& input0, const NdArrayRef& input1,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->BitwiseAnd(input0, input1);
             })
      .as(lhs.eltype());
}

NdArrayRef AndBBCorrelated::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                                 const NdArrayRef& rhs0,
                                 const NdArrayRef& rhs1) const {
  SPU_ENFORCE_EQ(lhs.shape(), rhs0.shape());
  SPU_ENFORCE_EQ(lhs.shape(), rhs1.shape());

  int64_t numel = lhs.numel();
  if (numel == 0) {
    return NdArrayRef(lhs.eltype(), {0});
  }

  return TiledDispatchTernaryFunc(
             ctx, lhs, rhs0, rhs1,
             [&](const NdArrayRef& input0, const NdArrayRef& input1,
                 const NdArrayRef& input2,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               auto [ret0, ret1] =
                   base_ot->CorrelatedBitwiseAnd(input0, input1, input2);

               NdArrayRef concat(ret0.eltype(), {ret0.numel() + ret1.numel()});

               std::memcpy(concat.data<std::byte>(), ret0.data<std::byte>(),
                           ret0.numel() * ret0.elsize());
               std::memcpy(
                   concat.data<std::byte>() + ret0.numel() * ret0.elsize(),
                   ret1.data<std::byte>(), ret1.numel() * ret1.elsize());
               return concat;
             })
      .as(lhs.eltype());
}
}  // namespace spu::mpc::cheetah
