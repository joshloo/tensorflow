// RUN: xla-opt %s -convert-triton-to-tritongpu='target=cuda:80'

module attributes {} {
  tt.func @gemm_fusion_dot_1_impl(%arg0: !tt.ptr<bf16>, %arg2: !tt.ptr<i16>) {
    %c0_i64 = arith.constant 0 : i64
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %14 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x16x!tt.ptr<bf16>>
    %24 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %30 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<32x2x!tt.ptr<i16>>
    %35:2 = scf.for %arg4 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %c0_i64, %arg8 = %cst_2) -> (i64, tensor<32x32xf32>)  : i32 {
      %50 = tt.splat %arg7 : i64 -> tensor<16xi64>
      %52 = tt.expand_dims %50 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
      %53 = tt.broadcast %52 : tensor<1x16xi64> -> tensor<32x16xi64>
      %55 = tt.addptr %14, %53 : tensor<32x16x!tt.ptr<bf16>>, tensor<32x16xi64>
      %56 = tt.load %55 : tensor<32x16x!tt.ptr<bf16>>
      %58 = tt.splat %arg7 : i64 -> tensor<32xi64>
      %60 = tt.expand_dims %58 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
      %61 = tt.broadcast %60 : tensor<32x1xi64> -> tensor<32x32xi64>
      %63 = tt.addptr %24, %61 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi64>
      %64 = tt.load %63 : tensor<32x32x!tt.ptr<bf16>>
      %66 = tt.splat %arg7 : i64 -> tensor<2xi64>
      %68 = tt.expand_dims %66 {axis = 0 : i32} : tensor<2xi64> -> tensor<1x2xi64>
      %69 = tt.broadcast %68 : tensor<1x2xi64> -> tensor<32x2xi64>
      %71 = tt.addptr %30, %69 : tensor<32x2x!tt.ptr<i16>>, tensor<32x2xi64>
      %72 = tt.load %71 : tensor<32x2x!tt.ptr<i16>>
      %74 = triton_gpu.sparse_dot %56, %64, %arg8, %72 : tensor<32x16xbf16> meta tensor<32x2xi16> * tensor<32x32xbf16> -> tensor<32x32xf32>
      scf.yield %arg7, %74 : i64, tensor<32x32xf32>
    }
    tt.return
  }
}