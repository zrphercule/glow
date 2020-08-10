/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "less_bfloat16Cases/0",
    "less_float16Cases/0",
    "less_int64Cases/0",
    "ResizeNearest_BFloat16/0",
    "ResizeNearest_BFloat16_outTy/0",
    "ResizeNearest_Float16/0",
    "ResizeNearest_Int16/0",
    "ResizeNearest_Float16_outTy/0",
    "ResizeNearest_Int16_outTy/0",
    "ResizeBilinear_BFloat16/0",
    "ResizeBilinear_BFloat16_outTy/0",
    "ResizeBilinear_Float16/0",
    "ResizeBilinear_Int16/0",
    "ResizeBilinear_Float16_outTy/0",
    "ResizeBilinear_Int16_outTy/0",
    "replaceNaN_BFloat16/0",
    "replaceNaN_Float16/0",
    "Logit_BFloat16/0",
    "Logit_Float16/0",
    "BFloat16Add/0",
    "FP16Add/0",
    "BFloat16Matmul/0",
    "FP16Matmul/0",
    "BroadCastMax/0",
    "BroadCastMin/0",
    "batchedPairwiseDotProduct/0",
    "batchedReduceAdd_BFloat16/0",
    "batchedReduceAdd_Float16/0",
    "batchedReduceZeroDimResult_BFloat16/0",
    "batchedReduceZeroDimResult_Float16/0",
    "batchedReduceAddWithAxis_BFloat16/0",
    "batchedReduceAddWithAxis_Float16/0",
    "ReluSimple_BFloat16/0",
    "ReluSimple_Float16/0",
    "PReluSimple_BFloat16/0",
    "PReluSimple_Float16/0",
    "GatherDataBFloat16IdxInt32/0",
    "GatherDataFloat16IdxInt32/0",
    "GatherDataBFloat16IdxInt64/0",
    "GatherDataFloat16IdxInt64/0",
    "GatherRangesDataBFloat16IdxInt32/0",
    "GatherRangesDataFloat16IdxInt32/0",
    "GatherRangesDataBFloat16IdxInt64/0",
    "GatherRangesDataFloat16IdxInt64/0",
    "BFloat16Transpose2Dims/0",
    "FP16Transpose2Dims/0",
    "Transpose3Dims_BFloat16/0",
    "Transpose3Dims_Float16/0",
    "ArithAdd_bfloat16_t/0",
    "ArithAdd_int32_t/0",
    "ArithAdd_int64_t/0",
    "ArithAdd_float16_t/0",
    "ArithSub_bfloat16_t/0",
    "ArithSub_int32_t/0",
    "ArithSub_int64_t/0",
    "ArithSub_float16_t/0",
    "ArithMul_bfloat16_t/0",
    "ArithMul_int32_t/0",
    "ArithMul_int64_t/0",
    "ArithMul_float16_t/0",
    "ArithMax_bfloat16_t/0",
    "ArithMax_int32_t/0",
    "ArithMax_int64_t/0",
    "ArithMax_float16_t/0",
    "ArithMin_bfloat16_t/0",
    "ArithMin_int32_t/0",
    "ArithMin_int64_t/0",
    "ArithMin_float16_t/0",
    "convTest_BFloat16/0",
    "convTest_Float16/0",
    "BFloat16Max/0",
    "FP16Max/0",
    "concatVectors_Int32/0",
    "concatVectors_BFloat16/0",
    "concatVectors_Float16/0",
    "concatVectorsRepeated_BFloat16/0",
    "concatVectorsRepeated_Int32/0",
    "concatVectorsRepeated_Float16/0",
    "sliceVectors_BFloat16/0",
    "sliceVectors_Float16/0",
    "sliceConcatVectors_BFloat16/0",
    "sliceConcatVectors_Float16/0",
    "ExpandDims_BFloat16/0",
    "ExpandDims_Float16/0",
    "Split_BFloat16/0",
    "Split_Float16/0",
    "BFloat16Splat/0",
    "Fp16Splat/0",
    "GroupConv3D/0",
    "NonCubicPaddingConv3D/0",
    "BFloat16AvgPool/0",
    "FP16AvgPool/0",
    "Int8AvgPool3D/0",
    "BFloat16AvgPool3D/0",
    "FP16AvgPool3D/0",
    "BFloat16AdaptiveAvgPool/0",
    "FP16AdaptiveAvgPool/0",
    "Int8AdaptiveAvgPool/0",
    "BFloat16MaxPool/0",
    "FP16MaxPool/0",
    "NonCubicKernelConv3D/0",
    "NonCubicKernelConv3DQuantized/0",
    "NonCubicStrideConv3D/0",
    "BFloat16BatchAdd/0",
    "FP16BatchAdd/0",
    "Sigmoid_BFloat16/0",
    "Sigmoid_Float16/0",
    "Swish_BFloat16/0",
    "Swish_Float16/0",
    "testBatchAdd_BFloat16/0",
    "testBatchAdd_Float16/0",
    "CumSum_BFloat16/0",
    "CumSum_Float/0",
    "CumSum_Float16/0",
    "CumSum_Int32/0",
    "CumSum_Int64/0",
    "CumSum_Exclusive/0",
    "CumSum_Reverse_BFloat16/0",
    "CumSum_Reverse/0",
    "CumSum_ExclusiveReverse/0",
    "CumSum_WithZeroes/0",
    "SparseLengthsSum_BFloat16/0",
    "SparseLengthsSum_Float16/0",
    "SparseLengthsSum_BFloat16_Int32/0",
    "SparseLengthsSum_Float16_Int32/0",
    "SparseLengthsSumI8/0",
    "SparseLengthsWeightedSum_1D_BFloat16/0",
    "SparseLengthsWeightedSum_1D_Float16/0",
    "SparseLengthsWeightedSum_2D_BFloat16/0",
    "SparseLengthsWeightedSum_2D_Float16/0",
    "EmbeddingBag_1D_BFloat16/0",
    "EmbeddingBag_1D_Float16/0",
    "EmbeddingBag_1D_BFloat16_End_Offset/0",
    "EmbeddingBag_1D_Float16_End_Offset/0",
    "EmbeddingBag_2D_BFloat16/0",
    "EmbeddingBag_2D_Float16/0",
    "EmbeddingBag_2D_BFloat16_End_Offset/0",
    "EmbeddingBag_2D_Float16_End_Offset/0",
    "SparseLengthsWeightedSumI8/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16_Int32/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat_Int32/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat_Int32/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16_Int32/"
    "0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_"
    "back/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_"
    "back2/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset_AccumFloat/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16_End_Offset/0",
    "EmbeddingBag_1D_Float_End_Offset_Partial/0",
    "EmbeddingBag_2D_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset_Partial/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Fused4Bit_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Fused4Bit_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert_FP32Accum/0",
    "SLWSTwoColumn_Float16_AccumFloat/0",
    "SparseToDense_Int64/0",
    "SparseToDenseMask1/0",
    "SparseToDenseMask2/0",
    "BoolReshape/0",
    "BFloat16Reshape/0",
    "FP16Reshape/0",
    "sliceReshape_BFloat16/0",
    "sliceReshape_Float16/0",
    "Flatten_BFloat16Ty/0",
    "Flatten_Float16Ty/0",
    "Bucketize/0",
    "BFloat16SoftMax/0",
    "FP16SoftMax/0",
    "BatchOneHotDataBFloat16/0",
    "BatchOneHotDataFloat/0",
    "BatchOneHotDataFloat16/0",
    "BatchOneHotDataInt64/0",
    "BatchOneHotDataInt32/0",
    "BatchOneHotDataInt8/0",
    "dotProduct1D_BFloat16/0",
    "dotProduct1D_Float16/0",
    "dotProduct2D_BFloat16/0",
    "dotProduct2D_Float16/0",
    "BatchBoxCox_Float/0",
    "BatchBoxCox_Large_BFloat16/0",
    "BatchBoxCox_Large_Float16/0",
    "BatchBoxCox_Medium_BFloat16/0",
    "BatchBoxCox_Medium_Float16/0",
    "BatchBoxCox_Small_BFloat16/0",
    "BatchBoxCox_Small_Float16/0",
    "ConvertFrom_BFloat16Ty_To_BFloat16Ty/0",
    "ConvertFrom_BFloat16Ty_To_FloatTy/0",
    "ConvertFrom_BFloat16Ty_To_Float16Ty/0",
    "ConvertFrom_BFloat16Ty_To_Int32ITy/0",
    "ConvertFrom_BFloat16Ty_To_Int64ITy/0",
    "ConvertFrom_BFloat16Ty_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_BFloat16Ty_To_FloatTy_AndBack/0",
    "ConvertFrom_BFloat16Ty_To_Float16Ty_AndBack/0",
    "ConvertFrom_BFloat16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_BFloat16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_FloatTy_To_BFloat16Ty/0",
    "ConvertFrom_FloatTy_To_Float16Ty/0",
    "ConvertFrom_FloatTy_To_Int32ITy/0",
    "ConvertFrom_FloatTy_To_Int64ITy/0",
    "ConvertFrom_Float16Ty_To_BFloat16Ty/0",
    "ConvertFrom_Float16Ty_To_FloatTy/0",
    "ConvertFrom_Float16Ty_To_Float16Ty/0",
    "ConvertFrom_Float16Ty_To_Int32ITy/0",
    "ConvertFrom_Float16Ty_To_Int64ITy/0",
    "ConvertFrom_Int32ITy_To_BFloat16Ty/0",
    "ConvertFrom_Int32ITy_To_Float16Ty/0",
    "ConvertFrom_Int64ITy_To_BFloat16Ty/0",
    "ConvertFrom_Int64ITy_To_FloatTy/0",
    "ConvertFrom_Int64ITy_To_Float16Ty/0",
    "ConvertFrom_BoolTy_To_BFloat16Ty/0",
    "ConvertFrom_BoolTy_To_FloatTy/0",
    "ConvertFrom_BoolTy_To_Float16Ty/0",
    "ConvertFrom_FloatTy_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_FloatTy_To_Float16Ty_AndBack/0",
    "ConvertFrom_FloatTy_To_Int32ITy_AndBack/0",
    "ConvertFrom_FloatTy_To_Int64ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_Float16Ty_To_FloatTy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Float16Ty_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_Int32ITy_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_Int32ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int64ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_Int32ITy_AndBack/0",
    "ConvertFusedToFusedFP16/0",
    "BasicDivNetFloatVsInt8/0",
    "BasicAddNetFloatVsBFloat16/0",
    "BasicAddNetFloatVsFloat16/0",
    "BasicSubNetFloatVsBFloat16/0",
    "BasicSubNetFloatVsFloat16/0",
    "BasicMulNetFloatVsBFloat16/0",
    "BasicMulNetFloatVsFloat16/0",
    "BasicDivNetFloatVsBFloat16/0",
    "BasicDivNetFloatVsFloat16/0",
    "BasicMaxNetFloatVsBFloat16/0",
    "BasicMaxNetFloatVsFloat16/0",
    "BasicMinNetFloatVsBFloat16/0",
    "BasicMinNetFloatVsFloat16/0",
    "Int16ConvolutionDepth10/0",
    "Int16ConvolutionDepth8/0",
    "BFloat16ConvolutionDepth10/0",
    "FP16ConvolutionDepth10/0",
    "BFloat16ConvolutionDepth8/0",
    "FP16ConvolutionDepth8/0",
    "FC_BFloat16/0",
    "FC_Float16/0",
    "Tanh_BFloat16/0",
    "Tanh_Float16/0",
    "Exp_BFloat16/0",
    "Exp_Float16/0",
    "rowwiseQuantizedSLWSTest/0",
    "SLSAllZeroLengths_Float16/0",
    "FusedRWQSLSAllZeroLengths_Float16/0",
    "RWQSLWSAllSame_Float16_AccumFP16/0",
    "RWQSLWSAllSame_Float16_AccumFP32/0",
    "SigmoidSweep_BFloat16/0",
    "SigmoidSweep_Float16/0",
    "TanHSweep_BFloat16/0",
    "TanHSweep_Float16/0",
    "RepeatedSLSWithPartialTensors/0",
    "GatherWithInt32PartialTensors/0",
    "GatherWithInt64PartialTensors/0",
    "ParallelBatchMatMul_BFloat16/0",
    "ParallelBatchMatMul_Float16/0",
    "ChannelwiseQuantizedGroupConvolution/0",
    "ChannelwiseQuantizedGroupConvolution3D/0",
    "ChannelwiseQuantizedGroupConvolutionNonZero/0",
    "CmpEQ_Int32/0",
    "SLWSAllLengthsOne_BFloat16_AccumFloat/0",
    "SLWSAllLengthsOne_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Fused4Bit_Float16_AccumFloat16/0",
    "LayerNorm_BFloat16/0",
    "FP16BatchNorm2D/0",
    "LayerNorm_Float16/0",
    "LayerNorm_Int8/0",
    "ChannelwiseQuantizedConv2D_NonZero_FloatBias/0",
    "DequantizeFRWQ_Float/0",
    "DequantizeFRWQ_Float16/0",
    "Abs_Int8QTy/0",
    "Neg_Int8QTy/0",
    "Floor_Int8QTy/0",
    "Sign_FloatTy/0",
    "Sign_Int8QTy/0",
    "Ceil_Int8QTy/0",
    "Round_Int8QTy/0",
    "Sqrt_Int8QTy/0",
    "Rsqrt_Int8QTy/0",
    "Reciprocal_Int8QTy/0",
    "Sin_Int8QTy/0",
    "Cos_Int8QTy/0",
    "rowwiseQuantizedFCTestAsymmetric_Int8_BiasFloat32/0",
    "rowwiseQuantizedFCTestSymmetric_Int8_BiasFloat32/0",
    "TestFP32Accumulator/0",
    "ROIAlign/0",
    "Asin_FloatTy/0",
    "Acos_FloatTy/0",
    "Atan_FloatTy/0",
    "Asin_Int8QTy/0",
    "Acos_Int8QTy/0",
    "Atan_Int8QTy/0",
    "BBoxTransform/0",
};
