/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Runs optimization passes on the given HLO module.
Status GpuCompiler::OptimizeHloModulePreFusion(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Save proto state before optimizations if we want a snapshot.
  if (DumpingEnabledForHloModule(*hlo_module)) {
    hlo_proto_ = absl::make_unique<HloProto>();
    *hlo_proto_->mutable_hlo_module() = hlo_module->ToProto();
  }

  const DebugOptions& debug_options = hlo_module->config().debug_options();

  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      // Run some IR cleanup passes before running the SPMD partitioning
      // passes.
      spmd_pipeline.AddInvariantChecker<HloVerifier>(
          /*layout_sensitive=*/false,
          /*allow_mixed_precision=*/false);
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<ZeroSizedHloElimination>();
      spmd_pipeline.AddPass<ConditionalCanonicalizer>();

      HloPassPipeline& spmd_simplify =
          spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

      AlgebraicSimplifierOptions options;
      options.set_replace_transpose_with_bitcast(false);
      options.set_enable_conv_operand_swap(false);
      // "slow" minmax means we propagate nan.
      options.set_minmax_propagate_nan(
          !debug_options.xla_gpu_enable_fast_min_max());
      spmd_simplify.AddPass<AlgebraicSimplifier>(options);

      spmd_simplify.AddPass<SortSimplifier>();
      spmd_simplify.AddPass<TupleSimplifier>();
      spmd_simplify.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);
      spmd_simplify.AddPass<GatherExpander>(
          GatherExpander::kEliminateSimpleGathers);
      spmd_simplify.AddPass<WhileLoopConstantSinking>();
      spmd_simplify.AddPass<WhileLoopSimplifier>();

      spmd_simplify.AddPass<ReshapeMover>();
      spmd_simplify.AddPass<HloConstantFolding>();
      spmd_simplify.AddPass<ConditionalSimplifier>();
      spmd_simplify.AddPass<HloDCE>();

      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true);
      spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
    } else {
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("optimization");
    pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                              /*allow_mixed_precision=*/false);
    pipeline.AddPass<AllToAllDecomposer>();

    OpExpanderPass::PatternExtraFilter upcaster_filter =
        [&](const HloInstruction* instr) {
          return !stream_exec->GetDeviceDescription()
                      .cuda_compute_capability()
                      .IsAtLeast(se::CudaComputeCapability::VOLTA) ||
                 !gpu::IsMatrixMultiplication(*instr);
        };

    pipeline.AddPass<OperandUpcaster>(upcaster_filter);
    pipeline.AddPass<ResultCaster>(upcaster_filter);

    // Expand random number generation.
    pipeline.AddPass<RngExpander>();
    pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

    // Comparison total order expander
    pipeline.AddPass<ComparisonExpander>();

    // Remove zero-sized HLO from the input so that other passes don't have to
    // handle it.
    pipeline.AddPass<ZeroSizedHloElimination>();

    if (debug_options.xla_gpu_deterministic_ops()) {
      // Scatter is nondeterministic, so eliminate all Scatters.
      pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
    } else {
      // Only Scatters unsupported on XLA:GPU are eliminated.
      pipeline.AddPass<GpuScatterExpander>();
    }
    // TODO(phawkins): replace QR and Eigh decompositions with calls to
    // cuSOLVER.
    pipeline.AddPass<QrExpander>();
    pipeline.AddPass<EighExpander>();

    pipeline.AddPass<DynamicIndexSplitter>();

    // TODO(b/64094172): make Call work on GPU instead of inlining.
    pipeline.AddPass<CallInliner>();

    pipeline.AddPass<DotDecomposer>();

    pipeline.AddPass<Convolution4DExpander>();

    // Expand the sort op to support stable sorting if required.
    pipeline.AddPass<StableSortExpander>();

    GpuBfloat16Support bf16(/*supports_matrix_multiplication=*/true,
                            stream_exec);
    pipeline.AddPass<BFloat16Normalization>(&bf16);

    pipeline.AddPass<BatchNormExpander>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true);

    pipeline.AddPass<LogisticExpander>(
        /*expansion_type=*/LogisticExpansionType::kExp);
    pipeline.AddPass<ConditionalCanonicalizer>();
    pipeline.AddPass<DynamicDimensionSimplifier>();
    auto dynamic_padder_options = DynamicPadderOptions();
    dynamic_padder_options.shape_check_mode =
        DynamicDimensionInference::ShapeCheckMode::kCompileTime;
    pipeline.AddPass<DynamicPadder>(dynamic_padder_options);

    // Build simplification pipeline.  The passes in here are run to a fixed
    // point.
    [&, &pipeline =
            pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
      pipeline.AddInvariantCheckerDebug<HloVerifier>(
          /*layout_sensitive=*/false,
          /*allow_mixed_precision=*/false);

      // BatchNormExpander can create zero-sized ops, so zero-sized HLO
      // elimination has to come after that pass.
      pipeline.AddPass<ZeroSizedHloElimination>();

      pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
      pipeline.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);

      AlgebraicSimplifierOptions options({}, ConvIsLowerable);
      // "slow" minmax means we propagate nan.
      options.set_minmax_propagate_nan(
          !debug_options.xla_gpu_enable_fast_min_max());

      // When transposes appear in a fusion node, we can easily adjust the
      // multi-dimensional index to create the one needed for the operand.
      // This is not as easy with bitcasts, because we don't have the
      // information readily available which dimensions are permuted. In
      // addition to that, if we have a transpose and a reshape next to each
      // other, they will both be replaced by a bitcast, and we replace
      // bitcast(bitcast) with one bitcast. This leads to having to
      // linearize and then delinearize the index.
      options.set_replace_transpose_with_bitcast(false);
      const se::Platform* platform = stream_exec->platform();
      if (platform->Name() == "ROCM") {
        // SwapConvOperands does not yet work on ROCM
        options.set_enable_conv_operand_swap(false);
      }
      pipeline.AddPass<AlgebraicSimplifier>(options);
      pipeline.AddPass<BitcastDtypesExpander>();
      // AlgebraicSimplifier may add contracting dimensions to a dot.
      pipeline.AddPass<DotDecomposer>();
      // Only merge "smallish" dots.  This threshold was not set carefully, but
      // so far we know that 1mb is too small.
      pipeline.AddPass<DotMerger>(/*max_size_to_merge=*/int64_t{16} << 20);
      pipeline.AddPass<SortSimplifier>();
      pipeline.AddPass<TupleSimplifier>();
      pipeline.AddPass<WhileLoopConstantSinking>();
      pipeline.AddPass<WhileLoopSimplifier>();

      // TODO(b/134075051): Re-enable after b/134075051 is fixed.
      // pipeline.AddPass<SliceSinker>();

      pipeline.AddPass<ReshapeMover>();
      pipeline.AddPass<HloConstantFolding>();
      pipeline.AddPass<ConditionalSimplifier>();
      pipeline.AddPass<RealImagExpander>();

      pipeline.AddPass<TransposeFolding>(
          [](const HloInstruction& dot,
             const TransposeFolding::OperandIndices& candidate_operands) {
            return IsMatrixMultiplication(dot)
                       ? candidate_operands
                       : TransposeFolding::OperandIndices{};
          });
      pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
      pipeline.AddPass<HloDCE>();
    }();

    // Run WhileLoopTripCountAnnotator at the end of the simplification
    // pipeline, before layout assignment and fusion.  This pass does some
    // pattern-matching on while bodies/conditions, and this is where the HLO is
    // "nicest".
    //
    // It's important that we don't make semantic changes (e.g. unrolling) to
    // any `while` loops after this point, because otherwise the trip-count
    // annotations added by this pass may not be correct after the
    // modifications.
    pipeline.AddPass<WhileLoopTripCountAnnotator>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  // Optimize collectives generated by SPMD partitioning. Enable these passes
  // otherwise as well so that all collectives can get these optimizations.
  {
    HloPassPipeline collectives_pipeline("collective-optimizations");
    collectives_pipeline.AddPass<AllReduceFolder>();
    collectives_pipeline.AddPass<ReduceScatterCreator>();
    collectives_pipeline.AddPass<AllReduceReassociate>();

    // Run algebraic simplifier to reshape(broadcast) into a broadcast when
    // the reshape is just adding a unit dimension. This will help with the
    // AllGatherBroadcastReorder pass.
    AlgebraicSimplifierOptions options;
    options.set_replace_transpose_with_bitcast(false);
    options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    options.set_minmax_propagate_nan(
        !debug_options.xla_gpu_enable_fast_min_max());

    collectives_pipeline.AddPass<AlgebraicSimplifier>(options);

    collectives_pipeline.AddPass<AllGatherBroadcastReorder>();
    TF_RETURN_IF_ERROR(collectives_pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes for convolution
  // canonicalization.
  TF_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(
      hlo_module, stream_exec, device_allocator));

  {
    // Run layout assignment in a separate pipeline from
    // "post-layout-assignment" because we want everything after layout
    // assignment to have a layout-sensitive invariant-checker, but
    // HloPassPipeline also runs its invariant checker before any passes are
    // run, meaning, the pipeline that contains layout assignment cannot contain
    // a layout-sensitive verifier!
    HloPassPipeline pipeline("layout assignment");
    // Layout assignment uses alias analysis, which requires the call graph to
    // be flattened.
    pipeline.AddPass<FlattenCallGraph>();
    ChannelLayoutConstraints layout_constraints;
    pipeline.AddPass<GpuLayoutAssignment>(
        hlo_module->mutable_entry_computation_layout(), stream_exec,
        &layout_constraints);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes after layout assignment.
  TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(hlo_module, stream_exec,
                                                     device_allocator));

  return Status::OK();
}

Status GpuCompiler::OptimizeHloModuleFusionRunPre(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  {
    HloPassPipeline fusion_pre("fusion_pre");
    // We try to split variadic ops with many parameters into several such ops
    // to avoid exceeding the parameter space.
    fusion_pre.AddPass<VariadicOpSplitter>();
    fusion_pre.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/true,
        /*allow_mixed_precision=*/false,
        LayoutAssignment::InstructionCanChangeLayout);
    TF_RETURN_IF_ERROR(fusion_pre.Run(hlo_module).status());
  }
  return Status::OK();
}

Status GpuCompiler::OptimizeHloModuleFusionRun(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator, bool may_duplicate) {
  {
    HloPassPipeline fusion_dry("fusion_dry");
    fusion_dry.AddPass<GpuInstructionFusion>(may_duplicate);
    TF_RETURN_IF_ERROR(fusion_dry.Run(hlo_module).status());
  }

  return Status::OK();
}

Status GpuCompiler::OptimizeHloModuleFusionRunPost(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  {
    HloPassPipeline fusion_post("fusion_post");
    fusion_post.AddPass<FusionMerger>();
    fusion_post.AddPass<GpuMultiOutputFusion>();
    fusion_post.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                           /*only_fusion_computations=*/true);
    fusion_post.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(fusion_post.Run(hlo_module).status());
  }
  return Status::OK();
}

Status GpuCompiler::OptimizeHloModulePostFusion(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  {
    HloPassFix<HloPassPipeline> horizontal_fusion("horizontal fusion");
    horizontal_fusion.AddPass<GpuHorizontalLoopFusion>();
    horizontal_fusion.AddPass<GpuHorizontalInputFusion>();
    // FusionBitcastLift must be after InstructionFusion, as it undoes
    // part of it.
    // TODO(b/209005695) Renable once the bug is fixed.
    // horizontal_fusion.AddPass<FusionBitcastLift>();
    horizontal_fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                                      /*only_fusion_computations=*/true);
    horizontal_fusion.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(horizontal_fusion.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("post-fusion optimization");
    pipeline.AddPass<AllGatherCombiner>(
        /*combine_threshold_in_bytes=*/1024 * 1024 * 1024,
        /*combine_threshold_count=*/256);
    pipeline.AddPass<AllReduceCombiner>(
        debug_options.xla_gpu_all_reduce_combine_threshold_bytes(),
        /*combine_threshold_count=*/256);
    pipeline.AddPass<ReduceScatterCombiner>(
        /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
        /*combine_threshold_count=*/256);

    if (debug_options.xla_gpu_all_reduce_contiguous()) {
      pipeline.AddPass<AllReduceContiguous>();
    }

    int32_t blueconnect_num_devices_per_host =
        debug_options.xla_gpu_all_reduce_blueconnect_num_devices_per_host();
    if (blueconnect_num_devices_per_host > 0) {
      pipeline.AddPass<AllReduceBlueConnect>(blueconnect_num_devices_per_host);
    }

    if (debug_options.xla_gpu_enable_async_all_reduce()) {
      AsyncCollectiveCreator::CollectiveCreatorConfig config;
      config.convert_all_reduce = [](const HloInstruction*) { return true; };
      pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));
    }

    pipeline.AddPass<CollectivesScheduleLinearizer>();

    // Now we allow replacing any transposes outside of fusions with bitcasts.
    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    options.set_minmax_propagate_nan(
        !debug_options.xla_gpu_enable_fast_min_max());
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<OptimizationBarrierExpander>();
    pipeline.AddPass<TupleSimplifier>();

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  return Status::OK();
}
