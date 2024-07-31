/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/tfrt/ifrt/ifrt_device_utils.h"

#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/mock.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::ReturnRef;
using ::tsl::testing::StatusIs;

static constexpr int kNumReplicas = 1;
static constexpr int kNumCoresPerReplica = 2;
// Intentionally have more devices than kNumReplicas * kNumCoresPerReplica for
// testing purposes.
static constexpr int kNumDevices = 4;
static constexpr int kDeviceIdOffset = 8;

class IfrtDeviceUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    devices_.reserve(kNumDevices);
    addressable_devices_.reserve(kNumDevices);
    for (int i = 0; i < kNumDevices; ++i) {
      devices_.push_back(std::make_unique<xla::ifrt::MockDevice>());
      EXPECT_CALL(*(devices_[i]), Attributes())
          .WillRepeatedly(ReturnRef(device_attributes_maps_[i]));
      EXPECT_CALL(*(devices_[i]), Id())
          .WillRepeatedly(Return(xla::ifrt::DeviceId(kDeviceIdOffset + i)));
      EXPECT_CALL(client_,
                  LookupDevice(xla::ifrt::DeviceId(kDeviceIdOffset + i)))
          .WillRepeatedly(Return(devices_[i].get()));

      addressable_devices_.push_back(devices_[i].get());
    };

    EXPECT_CALL(client_, addressable_devices())
        .WillRepeatedly(Return(addressable_devices_));

    // Default use the last two devices.
    xla::DeviceAssignment assignment(kNumReplicas, kNumCoresPerReplica);
    assignment(0, 0) = kDeviceIdOffset + 2;
    assignment(0, 1) = kDeviceIdOffset + 3;

    EXPECT_CALL(client_,
                GetDefaultDeviceAssignment(kNumReplicas, kNumCoresPerReplica))
        .WillRepeatedly(Return(assignment));
  }

  xla::ifrt::MockClient client_;
  std::vector<xla::ifrt::Device*> addressable_devices_;

  std::vector<std::unique_ptr<xla::ifrt::MockDevice>> devices_;
  std::vector<xla::ifrt::AttributeMap> device_attributes_maps_ = {
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({1, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(1)}}),
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({1, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(2)}}),
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({2, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(1)}}),
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({2, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(2)}}),
  };
};

TEST_F(IfrtDeviceUtilsTest, Basic) {
  std::vector<int> device_assignment_attr = {1, 0, 0, 2, 1, 0, 0, 1};
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(client_, kNumReplicas, kNumCoresPerReplica,
                             device_assignment_attr));
  EXPECT_THAT(devices_from_attribute,
              ElementsAre(addressable_devices_[1], addressable_devices_[0]));
}

TEST_F(IfrtDeviceUtilsTest, SeparateXCoordinates) {
  std::vector<int> device_assignment_attr = {1, 0, 0, 2, 2, 0, 0, 1};
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(client_, kNumReplicas, kNumCoresPerReplica,
                             device_assignment_attr));
  EXPECT_THAT(devices_from_attribute,
              ElementsAre(addressable_devices_[1], addressable_devices_[2]));
}

TEST_F(IfrtDeviceUtilsTest, EmptyDeviceAssignmentShallReturnDefault) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(client_, kNumReplicas, kNumCoresPerReplica,
                             std::nullopt));
  EXPECT_THAT(devices_from_attribute,
              ElementsAre(addressable_devices_[2], addressable_devices_[3]));
}

TEST_F(IfrtDeviceUtilsTest, MismatchCoordinatesShallFail) {
  std::vector<int> device_assignment_attr = {1, 0, 0, 2, 1, 0, 0, 3};
  auto status = GetAssignedIfrtDevices(client_, 1, 2, device_assignment_attr);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace

}  // namespace ifrt_serving
}  // namespace tensorflow
