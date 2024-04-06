// Copyright 2024 Ant Group Co., Ltd.
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

#include "seal/modulus.h"

#include "libspu/core/ndarray_ref.h"
namespace spu::mpc::cheetah {

class RingShareToPrimeShareLocalConvertor {
 public:
  explicit RingShareToPrimeShareLocalConvertor(
      size_t ring_width, const seal::Modulus& target_modulus);

  ~RingShareToPrimeShareLocalConvertor() = default;

  NdArrayRef Compute(const NdArrayRef& ring_share, int rank) const;

 private:
  size_t ring_width_;
  seal::Modulus target_modulus_;
  uint64_t offset_;  // -2^k mod p
};

}  // namespace spu::mpc::cheetah
