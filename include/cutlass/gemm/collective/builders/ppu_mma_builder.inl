#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"

#include "cutlass/detail/collective.hpp"

#include "cutlass/gemm/collective/builders/tile_shape_infer.inl"
#include "cutlass/gemm/config/gemm_operands.hpp"

#define ENABLE_AIU 1

namespace cutlass::gemm::collective {

namespace ppu_detail {

constexpr int ppu10000_smem_capacity_bytes = 262144;
// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  static_assert(carveout_bytes < ppu10000_smem_capacity_bytes, "epilogue carved out shm size should be smaller than total shm size");
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
  constexpr int compute_stages = (CapacityBytes - carveout_bytes) / stage_bytes;
  constexpr int out_stages = min(compute_stages, Int<5>{});
  return out_stages;
}
} // namespace ppu_detail
namespace detail {

///////////////////////////////////////////////////////////////////////////////
#if ENABLE_AIU
// ContigShape_: describes WHAT MAKES UP the AIU's 32-byte contiguous run. void (default) = the run is pure K, i.e.
// Shape<Block_K> -- byte-identical to the original derivation for every existing config (proved for int4 TK64/TK128,
// int2 TK128/TK256, int1 TK256 x MN64/MN128). An N-FOLD instead passes Shape<Int<FoldF>, Int<TK>>, so the 32B rule is
// satisfied by (FoldF N-columns x TK) and every derived quantity (AiuContElemSize, InstNum, bits_per_aiu, swzl CUBE_W)
// follows automatically -- no manual Block_K/Block_MN doubling to keep in sync across four places.
template <
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap,
  typename ContigShape_ = void
> struct MixGemm_AIU_Operand;

namespace aiu_detail {
// contiguous element count of the run: size(ContigShape) when given, else Block_K
template <class Block_K, class ContigShape_> struct contig_elems {
  static constexpr int value = cute::size(ContigShape_{});
};
template <class Block_K> struct contig_elems<Block_K, void> {
  static constexpr int value = Block_K{};
};
} // namespace aiu_detail

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  Element,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<Element>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_trans: block contiguous size should be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<Element>::value * 8>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContElemSize{} * sizeof_bits<Element>::value;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  cutlass::int4b_t,
  false,
  Block_MN,
  Block_K,
  Swap    // NO trailing comma: a trailing comma in a template-argument-list is ill-formed C++. clang (hence hgcc)
          // accepts it as an extension; nvcc's EDG front end rejects it, which made this specialization -- and
          // therefore CollectiveMma and EVERY collective downstream of it -- fail to instantiate under the local
          // nvcc front-end gate. Two static errors reached the box behind that one character.
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::int4b_t>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::int4b_t>::value * 8>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::int4b_t, false>;     // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::int4b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 2, Swap, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

// W2A16 base plane: uint2b_t swzl operand. Mirrors the int4b_t spec above; only difference is 4 int2/byte
// (int4 has 2/byte) -> the int8-typed swzl element count is AiuContElemSize/4 (int4 uses /2). The resulting
// smem->reg fragment order is NOT the same as int4's -> its converter reshuffle must be probed on ppu001
// (see fast_numeric_conversion_for_mix_gemm.h uint2b_t wide converter TODO).
template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  cutlass::uint2b_t,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::uint2b_t>::value / 8;   // Block_K/4 bytes
  static_assert(BlockContSize % 32 == 0, "aiu w2: block_k*2/8 must be multiple of 32B (block_k % 128 == 0)");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu w2: 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::uint2b_t>::value * 8>;      // AiuContByteSize*4
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::uint2b_t, false>;            // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::uint2b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 4, Swap, false, InstNum>;  // 4 int2/byte
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

// W1A16 base plane: uint1b_t swzl operand. Mirrors uint2b_t; 8 int1/byte (int2 has 4/byte) -> the int8-typed
// swzl element count is AiuContElemSize/8. Block_K*1/8 must be %32==0 -> Block_K % 256 == 0.
template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
  cutlass::uint1b_t,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::uint1b_t>::value / 8;   // Block_K/8 bytes
  static_assert(BlockContSize % 32 == 0, "aiu w1: block_k*1/8 must be multiple of 32B (block_k % 256 == 0)");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu w1: 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::uint1b_t>::value * 8>;      // AiuContByteSize*8
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::uint1b_t, false>;            // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::uint1b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 8, Swap, false, InstNum>;  // 8 int1/byte
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

#endif

template <typename Arch,
          typename ElementA,
          typename ElementB,
          typename ElementAccumulator,
          typename TileShape_MNK,
          typename ClusterShape_MNK,
          typename PermutionK_ = void
          >
struct get_tiled_mma {
  using MmaInst = typename config::GetMmaInst<Arch, ElementA, ElementB, ElementAccumulator>::type;

  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  // User can configure custom warp tile shape through ClusterShape_MNK
  static constexpr bool CustomWarpShape = cute::get<0>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<1>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<2>(ClusterShape_MNK{}) != 1;
  static constexpr auto WarpShapeStage = MapBlockShapeToWarpShapeStage<Arch, ElementA, blockM, blockN, blockK>();

  static constexpr int InstM = cute::get<0>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstN = cute::get<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstK = cute::get<2>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  static constexpr int warpM = max(Int<InstM>{}, CustomWarpShape ? cute::get<0>(ClusterShape_MNK{}) : cute::get<0>(WarpShapeStage));
  static constexpr int warpN = max(Int<InstN>{}, CustomWarpShape ? cute::get<1>(ClusterShape_MNK{}) : cute::get<1>(WarpShapeStage));

  using WarpOnM = Int<blockM / warpM>;
  using WarpOnN = Int<blockN / warpN>;

  using PermutionK = cute::conditional_t<cute::is_void_v<PermutionK_>, Int<InstK>, PermutionK_>;
  static_assert(PermutionK{} % InstK == 0, "PermutionK must be multiple of InstK.");

  using TiledMma = cute::conditional_t<cute::is_void_v<PermutionK_>,
                      TiledMMA<MMA_Atom<MmaInst>,
                              cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>,
                      TiledMMA<MMA_Atom<MmaInst>,
                              cute::Layout<Shape<WarpOnM, WarpOnN, _1>>,
                              Tile<Int<blockM / warpM * InstM>, Int<blockN / warpN * InstN>, PermutionK>>>;
};

} // namespace detail


// AIU GEMM
template <
  typename Arch,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelScheduleAuto> ||
       cute::is_same_v<KernelScheduleType, KernelMultistage> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelTma> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>)>
> {
  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = typename detail::get_tiled_mma<Arch, MmaElementA, MmaElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      MmaElementA, MmaElementB, TileShape_MNK>(StageCountType{});
#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiu<PipelineStages, PPUKernelScheduleType>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;


  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUCpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<MmaElementA, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<MmaElementB, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    ElementA, TagToStrideA_t<GmemLayoutA>,
    ElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB   // B
  >;

};

// AIU GEMM with scale (for a8w8 block-wise quant)
template <
  typename Arch,
  class ElementA,
  class GmemLayoutPairA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutPairB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutPairA,
    AlignmentA,
    ElementB,
    GmemLayoutPairB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<(cute::is_same_v<KernelScheduleType, KernelAiuMultistageWithScale>
      || cute::is_same_v<KernelScheduleType, KernelAiuMultistageWithBlockWiseScale>)
  >
> {
  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairA{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairA{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairB{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairB{}))>;
  static_assert(cute::depth(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 2 and
                cute::depth(cute::remove_pointer_t<GmemLayoutSFBTag>{}) == 2,
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{}) ==
                size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{}),
      "SFA and SFB must have equivalent SF vector sizes along K");
  static_assert(sizeof_bits<ElementA>::value == 8, "Only int8 or fp8 is supported for ElementA at KernelAiuMultistageWithScale.");
  static_assert(sizeof_bits<ElementB>::value == 8, "Only int8 or fp8 is supported for ElementB at KernelAiuMultistageWithScale.");

  static constexpr auto ScaleGranularityM = size<0,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});
  static constexpr auto ScaleGranularityN = size<0,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});
  static constexpr auto ScaleGranularityK = size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});

  using ElementScale = float;

  using TiledMma = typename detail::get_tiled_mma<Arch, ElementA, ElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});


#if ENABLE_AIU
  using DispatchPolicy = MainloopWithScalePPUAiu<PipelineStages, KernelScheduleType>;
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  static constexpr int ScaleMsPerTile = cute::ceil_div(Int<blockM>{}, Int<ScaleGranularityM>{});
  static constexpr int ScaleNsPerTile = cute::ceil_div(Int<blockN>{}, Int<ScaleGranularityN>{});
  static constexpr int ScaleKsPerTile = cute::ceil_div(Int<blockK>{}, Int<ScaleGranularityK>{});

  static constexpr int MaxAiuContElemSize = 128 / (sizeof_bits<ElementScale>::value / 8);     // 32
  static constexpr int MinAiuContElemSize = 32 / (sizeof_bits<ElementScale>::value / 8);      // 8

  static_assert(blockK > ScaleGranularityK ? (blockK % ScaleGranularityK) == 0 : (ScaleGranularityK % blockK) == 0,
              "block scaling granularity must evenly divide tile shape along K.");

  static constexpr bool TransA = platform::is_same<GmemLayoutATag, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutBTag, cutlass::layout::ColumnMajor>::value ? false : true;

  static constexpr bool TransSFA = is_static<decltype(stride<1>(GmemLayoutSFATag{}))>::value ? false : true;
  static constexpr bool TransSFB = is_static<decltype(stride<1>(GmemLayoutSFBTag{}))>::value ? false : true;

  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;

  static constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSize) : ScaleNsPerTile;
  static constexpr int SFBTileK = TransSFB ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);

#if 1
  static constexpr int SFATileM = TransSFA ? cute::max(ScaleMsPerTile, MinAiuContElemSize) : ScaleMsPerTile;
  static constexpr int SFATileK = TransSFA ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);
  using DefaultOperandSFA = config::DefaultGemm_AIU_Operand<Arch, ElementScale, TransSFA, Int<SFATileM>, Int<SFATileK>, false, 1, false>;
#else
  // TransSFA == true, scale_M % ScaleMsPerTile ==0, fold
  using DefaultOperandSFA = config::DefaultGemm_AIU_Operand<
      Arch, ElementScale, TransSFA, Int<MaxAiuContElemSize>, Int<ScaleMsPerTile / MaxAiuContElemSize>, false, 1, false>;
#endif
  using DefaultOperandSFB = config::DefaultGemm_AIU_Operand<Arch, ElementScale, TransSFB, Int<SFBTileN>, Int<SFBTileK>, true, 1, false>;

#else
  using DispatchPolicy = MainloopPPUCpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = cute::identity;
  using TransformB = cute::identity;

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // scaleA
  using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom; // M, K
  using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;

  // scaleB
  using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom; // N, K
  using GmemTiledCopySFB = typename DefaultOperandSFB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    ElementA, cute::tuple<TagToStrideA_t<GmemLayoutATag>, TagToStrideA_t<GmemLayoutSFATag>>,
    ElementB, cute::tuple<TagToStrideB_t<GmemLayoutBTag>, TagToStrideB_t<GmemLayoutSFBTag>>,
    TiledMma,
    cute::tuple<GmemTiledCopyA, GmemTiledCopySFA>,
    cute::tuple<SmemLayoutAtomA, SmemLayoutAtomSFA>,
    SmemCopyAtomA, TransformA,     // A
    cute::tuple<GmemTiledCopyB, GmemTiledCopySFB>,
    cute::tuple<SmemLayoutAtomB, SmemLayoutAtomSFB>,
    SmemCopyAtomB, TransformB      // B
  >;
};

// AIU Mixed GEMM
template <
  typename Arch,
  class ElementPairA_,
  class GmemLayoutA_,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutB_,
  int AlignmentB,
  class ElementAccumulator,
  class TileShapePair_,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementPairA_,
    GmemLayoutA_,
    AlignmentA,
    ElementPairB_,
    GmemLayoutB_,
    AlignmentB,
    ElementAccumulator,
    TileShapePair_,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputPerCol> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs128> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs64> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs32> ||
       (fold_schedule_traits<KernelScheduleType>::FoldF > 0))>   // N-FOLD: KernelAiuFold<FoldF, Base>
> {
private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementPairA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementPairB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementPairA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementPairB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementPairA_>::value && !cute::is_tuple<ElementPairB_>::value;

public:
  using TileShape_MNK = detail::deduce_mixed_width_dtype_t<0, TileShapePair_>;
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementPairA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementPairB_>;
  static_assert(cute::is_tuple<ElementPairA_>::value ^ cute::is_tuple<ElementPairB_>::value ||
               (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value)),
    "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  using ElementPairA = cute::conditional_t<IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementPairA_>;
  using ElementPairB = cute::conditional_t<!IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementPairB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

  using ElementMma = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using RealInternalElementA = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using RealInternalElementB = cute::conditional_t<IsATransformed, ElementA, ElementB>;

  // currently only support a16w8 / a16w4 mix gemm
  // static_assert(IsATransformed, "currently only A is supported for quantization.");
  static_assert(sizeof_bits<RealInternalElementA>::value == 16 && (sizeof_bits<RealInternalElementB>::value == 8 || sizeof_bits<RealInternalElementB>::value == 4 || sizeof_bits<RealInternalElementB>::value == 2 || sizeof_bits<RealInternalElementB>::value == 1),
    "currently only support a16w8 / a16w4 / a16w2 / a16w1 mix gemm");
  // For fp32 types, map to tf32 MMA value type
  // using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  // using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // PermutionK = the K span one B swzl copy step delivers = 32B worth of the packed element (int4 64 / int2 128 /
  // int1 256). Under an N-FOLD that 32B run is FoldF N-cols x blockK each, so the K span the MMA sees is only
  // blockK -- using the unfolded value would exceed TileShape.K and index past the tile.
  // FOLD: the fragment must have ORDINARY N x K register semantics (the fold lives only in the load layer), so the
  // K-permutation is TileShape.K when folding -- NOT the 32B-run span, which would keep the fragment in the folded
  // (N/FoldF) x (FoldF*K) form and force a 2-pass mainloop.
  // (e) ONE definition, shared with the offline generators -- see MixGemmMmaPermK in
  // fast_numeric_conversion_for_mix_gemm.h for why restating it broke a folded plane.
  static constexpr int MmaPermK =
      cutlass::MixGemmMmaPermK<sizeof_bits<RealInternalElementB>::value,
                               cute::get<2>(TileShape_MNK{}),
                               (fold_schedule_traits<KernelScheduleType>::FoldF > 0 ? 2 : 1)>::value;
  using TiledMma = typename detail::get_tiled_mma<
        Arch, ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
        Int<MmaPermK>>::TiledMma;

  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      ElementMma, ElementMma, TileShape_MNK>(StageCountType{});

  // currently only support k-major
  static_assert(cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajor> || cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajorInterleaved<256>>,
      "invalid GmemLayoutA, currently only support k-major or k256-major");
  static_assert(cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajor> || cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajorInterleaved<256>>,
      "invalid GmemLayoutB, currently only support k-major or k256-major");

  using kContinousA = cute::conditional_t<cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajorInterleaved<256>>, Int<256>, Int<1>>;
  using kContinousB = cute::conditional_t<cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajorInterleaved<256>>, Int<256>, Int<1>>;
  using kContinous = cute::conditional_t<IsATransformed, kContinousA, kContinousB>;
  // ---- B BIT-PLANE CONCAT: a 4th member in the B element tuple (tuple<ElementB,Scale,Zero,PlaneB2>) routes to
  // the dedicated TWO-plane mainloop. Absent => void => everything below degenerates to the single-plane build
  // BIT-IDENTICALLY (same policy, same atoms, no BPlanes wrapper).
  // NOTE PermutionK further down uses sizeof_bits<RealInternalElementB>, i.e. the LOW plane -- which is exactly
  // right: the low plane drives the main swzl and tCrB_mma; the high plane only feeds extra bits to the converter.
  using PlaneB2 = detail::deduce_mixed_width_dtype_t<3, ElementPairB>;
  static constexpr bool HasPlane2 = !cute::is_void_v<PlaneB2>;

  // N-FOLD: KernelScheduleType may be KernelAiuFold<FoldF, Base>. Extract FoldF + the underlying (group-size) base
  // schedule; when folding, route to MainloopPPUAiuFold and give the B operand a folded Block_K (below).
  static constexpr int FoldF = fold_schedule_traits<KernelScheduleType>::FoldF;
  static constexpr bool HasFold = FoldF > 0;
  using BaseSchedule = typename fold_schedule_traits<KernelScheduleType>::Base;   // == KernelScheduleType if no fold

  // HasPlane2 must WIN over HasFold. It used to be the other way round, which meant a 2-plane build whose LOW plane
  // needs a fold (int2 at Block_K=64 -> F1=2) was routed to the single-plane fold collective and plane 2 was silently
  // dropped. FoldF does not need to enter the 2-plane dispatch policy: each plane's fold factor is readable off its own
  // SmemLayoutAtom, which the builder already sizes folded (BFoldBlockK / the per-plane B2 block), exactly as the
  // collective already does for P2Fold. BaseSchedule keeps the group-size schedule either way, and MmaPermK above is
  // already the fold rule whenever FoldF > 0, which is what a folded low plane needs.
  using DispatchPolicy = cute::conditional_t<HasPlane2,
      MainloopPPUAiuMixedInput2Plane<PipelineStages, kContinous, BaseSchedule>,
      cute::conditional_t<HasFold,
          MainloopPPUAiuFold<PipelineStages, kContinous, (HasFold ? FoldF : 2), BaseSchedule>,
          MainloopPPUAiuMixedInput<PipelineStages, kContinous, BaseSchedule>>>;

  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  // N-FOLD: the B plane's AIU contiguous run folds FoldF adjacent N-cols x blockK each, so its operand Block_K =
  // FoldF*blockK (=> AiuContElemSize = FoldF*blockK, reusing a validated config, e.g. int2 blockK=64 FoldF=2 => 128
  // == int2@TK128). A stays blockK. The collective's fold-in-N SmemLayoutB then presents this as (FoldF*Ng, blockK).
  static constexpr int BFoldBlockK = (HasFold ? FoldF : 1) * blockK;
  // ...and the PHYSICAL row count halves/quarters correspondingly: folding FoldF N-columns into one contiguous run
  // means the B tile physically has blockN/FoldF rows of FoldF*blockK each (same total bytes). Folding only K while
  // leaving Block_MN=blockN makes the swzl atom address a 2x-too-large tile per stage -> "TSM out of range" at
  // runtime (observed: tsm.ld.swzl stepping 0x800 through a 0x400-per-stage buffer).
  static constexpr int BFoldBlockN = blockN / (HasFold ? FoldF : 1);
  // (MOEG_FOLD_DEBUG dump removed -- it confirmed fold_dbg<64,64,64,2,32,128>: builder params are CORRECT,
  //  i.e. B operand gets Block_MN=32 / Block_K=128 -> swzl CUBE <32,32> -> 1024B/stage, matching SmemLayoutB.)
  using DefaultOperandA = detail::MixGemm_AIU_Operand<RealInternalElementA, false, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::MixGemm_AIU_Operand<RealInternalElementB, false, Int<BFoldBlockN>, Int<BFoldBlockK>, true>;
#elif 0 // async_cp not work now
  static_assert(false, "async_cp not work now");
  using DispatchPolicy = MainloopPPUAiuMixedInput<PipelineStages, kContinous, KernelScheduleType>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    RealInternalElementA, GmemLayoutA, cute::conditional_t<IsATransformed, AlignmentA, AlignmentB>, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    RealInternalElementB, GmemLayoutB, cute::conditional_t<IsATransformed, AlignmentB, AlignmentA>, 32>;
#endif
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B plane 0 = the LOW plane
  using SmemLayoutAtomB0 = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB0 = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB0 = typename DefaultOperandB::GmemTiledCopy;

  // B plane 1 = the HIGH plane. Both planes share ONE tile (so the tile is bounded below by the SPARSEST plane's
  // AIU 32B minimum: int1 => Block_K>=256, int2 => >=128); only the element width differs, so plane 1's AIU/swzl
  // config comes out with the matching 2x/4x-smaller byte extent automatically. The fallback element keeps this
  // well-formed (and unused) in single-plane builds.
  // NOTE Block_K here must ALSO be the folded one (BFoldBlockK): even in single-plane builds this type gets
  // instantiated (it feeds the unused BPlanes fallback), so with a fold the plain blockK would give a sub-32B
  // contiguous run and trip MixGemm_AIU_Operand's `BlockContSize % 32 == 0` static_assert.
  // PER-PLANE N-FOLD. The fold factor is a per-plane quantity -- F_p = contig_p >= 32 ? 1 : 32/contig_p with
  // contig_p = Block_K * bits_p / 8 -- and plane 2 is the SPARSER plane, so at the same Block_K its contiguous run is
  // 2x/4x smaller and can fall below the AIU's 32 B minimum. Giving both planes ONE fold factor is exactly what pinned
  // the 2-plane path to Block_K >= 256: the only K where int2 and int1 both reach 32 B unfolded. Plane 2 now folds the
  // EXTRA amount it needs on top of plane 1's (BFoldBlockK already carries plane 1's), so Q3 reaches Block_K = 128
  // (int2 F=1, int1 F=2) and 64 (F=2 / F=4). Derived (fold_derivation/l42_2plane_fold.cu): every folded configuration
  // comes out with P2_DIV = 1, simpler than today's 2, and at Block_K=64 the shared logical mma fragment is the same
  // ((2,2,2),4,4):((1,2,4),32,8) as the single-plane int1 config that measures 63.7%.
  using P2Elem = cute::conditional_t<HasPlane2, PlaneB2, RealInternalElementB>;
  static constexpr int P2Contig = BFoldBlockK * cutlass::sizeof_bits<P2Elem>::value / 8;  // bytes AFTER plane 1's fold
  static constexpr int P2Fold   = P2Contig >= 32 ? 1 : 32 / P2Contig;   // the extra fold plane 2 needs
  static_assert(P2Contig * P2Fold >= 32 || !HasPlane2,
      "plane 2 cannot reach the AIU 32 B contiguous minimum at this Block_K even folded -- raise Block_K");
  static_assert(BFoldBlockN % P2Fold == 0 || !HasPlane2, "plane 2's fold must divide Block_N");
  using DefaultOperandB2 = detail::MixGemm_AIU_Operand<
      P2Elem, false, Int<BFoldBlockN / P2Fold>, Int<BFoldBlockK * P2Fold>, true>;

  // Both planes' atoms ride the EXISTING single template params (CollectiveMma's parameter list is fixed by its
  // primary template). collective::BPlanes is the marker -- NOT cute::is_tuple, since a cute Layout is itself
  // tuple-like and would false-positive on SmemLayoutAtomB.
  using SmemLayoutAtomB = cute::conditional_t<HasPlane2,
      collective::BPlanes<SmemLayoutAtomB0, typename DefaultOperandB2::SmemLayoutAtom>, SmemLayoutAtomB0>;
  using SmemCopyAtomB = cute::conditional_t<HasPlane2,
      collective::BPlanes<SmemCopyAtomB0, typename DefaultOperandB2::SmemCopyAtom>, SmemCopyAtomB0>;
  using GmemTiledCopyB = cute::conditional_t<HasPlane2,
      collective::BPlanes<GmemTiledCopyB0, typename DefaultOperandB2::GmemTiledCopy>, GmemTiledCopyB0>;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShapePair_,
    ElementPairA, TagToStrideA_t<GmemLayoutA>,
    ElementPairB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};

// AIU GEMM for Batch Array
template <
  typename Arch,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative>)>
> {

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;
  using TiledMma = typename detail::get_tiled_mma<Arch, MmaElementA, MmaElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      MmaElementA, MmaElementB, TileShape_MNK>(StageCountType{});

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiuBatchArray<PipelineStages>;
  static constexpr bool TransA = platform::is_same<typename TagToStrideA<GmemLayoutA>::tag, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<typename TagToStrideB<GmemLayoutB>::tag, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUCpAsync<PipelineStages>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<MmaElementA, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<MmaElementB, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    ElementA, TagToStrideA_t<GmemLayoutA>,
    ElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB   // B
  >;

};

// AIU GEMM for StreamK
template <
  typename Arch,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelAiuMultistageStreamK>)>
> {

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;
  using ElementMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;

  // TODO: 32x64x64 warp tile would make the barrier wait in ppu_tile_scheduler_stream_k.hpp hang
  //       but 64x64x64 warp tile is OK. This kind of bug already existed when Xiaohui first enabled Stream-K, would investigate later.
  #if 0
  using TiledMma = typename detail::get_tiled_mma<Arch, ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  #else
  using MmaInst = typename config::GetMmaInst<Arch, ElementMma, ElementMma, ElementAccumulator>::type;
  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_32, _32, _16>>; // 2x1x1 value group for 16x16x16 MMA and LDSM
  #endif

  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      ElementMma, ElementMma, TileShape_MNK>(StageCountType{});

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiu<PipelineStages, KernelAiuMultistageStreamK>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUCpAsync<PipelineStages>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementMma, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementMma, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    MmaElementA, TagToStrideA_t<GmemLayoutA>,
    MmaElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB   // B
  >;

};

} // namespace cutlass::gemm::collective
