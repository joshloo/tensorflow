/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

llvm::SmallVector<bool, 2> ResolveWindowReversal(
    const int64_t num_spatials,
    std::optional<mlir::DenseElementsAttr> opt_reversals) {
  if (!opt_reversals.has_value()) {
    return llvm::SmallVector<bool, 2>(num_spatials, false);
  }
  auto reversals = opt_reversals.value();
  if (reversals.isSplat()) {
    return llvm::SmallVector<bool, 2>(num_spatials,
                                      reversals.getSplatValue<bool>());
  }
  return llvm::SmallVector<bool, 2>(reversals.getValues<bool>());
}

ConvView::ConvView(mhlo::ConvolutionOp op)
    : input_layout_(
          Layout{op.getDimensionNumbers().getInputBatchDimension(),
                 op.getDimensionNumbers().getInputFeatureDimension(),
                 op.getDimensionNumbers().getInputSpatialDimensions()}),
      kernel_layout_(
          Layout{op.getDimensionNumbers().getKernelInputFeatureDimension(),
                 op.getDimensionNumbers().getKernelOutputFeatureDimension(),
                 op.getDimensionNumbers().getKernelSpatialDimensions()}),
      output_layout_(
          Layout{op.getDimensionNumbers().getOutputBatchDimension(),
                 op.getDimensionNumbers().getOutputFeatureDimension(),
                 op.getDimensionNumbers().getOutputSpatialDimensions()}),
      input_shape_(
          llvm::SmallVector<int64_t, 4>(op.getLhs().getType().getShape())),
      kernel_shape_(
          llvm::SmallVector<int64_t, 4>(op.getRhs().getType().getShape())),
      output_shape_(
          llvm::SmallVector<int64_t, 4>(op.getResult().getType().getShape())),
      batch_group_count_(op.getBatchGroupCount()),
      feature_group_count_(op.getFeatureGroupCount()),
      element_type_(op.getLhs().getType().getElementType()) {
  const int64_t num_spatials = InputLayout().NumSpatials();

  strides_ = ResolveStridesOrDilations(num_spatials, op.getWindowStrides());

  input_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getLhsDilation());
  kernel_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getRhsDilation());

  padding_ = ResolvePadding(num_spatials, op.getPadding());

  window_reversal_ =
      ResolveWindowReversal(num_spatials, op.getWindowReversal());
}

}  // namespace mlir::odml
