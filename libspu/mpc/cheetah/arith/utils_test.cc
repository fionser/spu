// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/arith/utils.h"

#include "gtest/gtest.h"
#include "seal/seal.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace spu::mpc::cheetah::test {

template <typename T>
T makeMask(size_t bits) {
  if (sizeof(T) * 8 == bits) {
    return -1;
  }
  return static_cast<T>(1) << bits;
}

class UtilsTest : public ::testing::Test {};

TEST_F(UtilsTest, RingToPrime) {
  auto ft = FM64;
  auto prime = seal::CoeffModulus::Create(8192, {60})[0];
  int64_t n = 1000000;

  for (size_t ring_w = 32; ring_w <= 64; ring_w += 8) {
    RingShareToPrimeShareLocalConvertor conv(ring_w, prime);
    NdArrayRef msg = ring_rand(ft, {n});
    // |msg| < 2^30
    ring_rshift_(msg, 64 - 30);

    NdArrayRef rnd0 = ring_rand(ft, {n});
    NdArrayRef rnd1 = ring_sub(msg, rnd0);

    auto h0 = conv.Compute(rnd0, 0);
    auto h1 = conv.Compute(rnd1, 1);

    NdArrayView<uint64_t> xmsg(msg);
    NdArrayView<uint64_t> xh0(h0);
    NdArrayView<uint64_t> xh1(h1);

    int64_t err_cnt = 0;
    for (int64_t i = 0; i < n; ++i) {
      auto got = seal::util::add_uint_mod(xh0[i], xh1[i], prime);
      err_cnt += (got == xmsg[i] ? 0 : 1);
    }

    double expected_ratio = 0.5 * std::pow(2., 30 - static_cast<int>(ring_w));
    double got_ratio = err_cnt * 1. / n;
    ASSERT_NEAR(expected_ratio, got_ratio, std::max(expected_ratio, got_ratio));
  }
}

}  // namespace spu::mpc::cheetah::test
