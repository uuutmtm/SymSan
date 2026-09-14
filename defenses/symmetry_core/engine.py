"""Applies the symmetry transforms to a model and verifies equivalence."""

import time
import torch
import torch.nn as nn
from typing import Optional

from .architecture import ArchitectureAnalyzer, ArchitectureInfo, print_architecture_summary
from .transforms import SymmetryTransforms
from .verification import EquivalenceVerifier, print_verification_results


class SymmetryDefenseEngine:
    """
    Main entry point for the symmetry-based defense.

    Usage:
        engine = SymmetryDefenseEngine(model, tokenizer, seed=42)
        model = engine.run()
    """

    def __init__(self, model: nn.Module, tokenizer=None,
                 seed: int = 42, scale_range: float = 0.05,
                 verify: bool = True, verbose: bool = True):
        """
        Args:
            model: HuggingFace CausalLM model (modified in-place).
            tokenizer: Tokenizer for equivalence verification.
            seed: Random seed for reproducibility.
            scale_range: Range for scaling factors [1-r, 1+r].
            verify: Whether to run equivalence verification.
            verbose: Whether to print progress to terminal.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.scale_range = scale_range
        self.do_verify = verify
        self.verbose = verbose
        # Filled by run(): machine-readable summary of the last execution.
        self.last_report: Optional[dict] = None

    def run(self) -> nn.Module:
        """Execute the full defense pipeline. Returns the transformed model."""
        t_start = time.time()

        # Set random seed
        torch.manual_seed(self.seed)

        # ====== Stage 1: Architecture Analysis ======
        if self.verbose:
            self._print_header("Stage 1: Architecture Analysis")

        analyzer = ArchitectureAnalyzer(self.model)
        info = analyzer.analyze()

        # Use the unwrapped LM model for verification (excludes vision tower etc.)
        lm_model = analyzer.lm_model

        if self.verbose:
            print_architecture_summary(info)

        # Validate that we found the essential components
        self._validate_architecture(info)

        # Untie word embeddings so that the final-norm scaling pair
        # (gamma_final <-> lm_head) applies on tied models as well.
        self._maybe_untie_lm_head(info)

        # ====== Stage 2: Pre-Transform Snapshot ======
        # Prefer the full wrapper for verification when only it exposes
        # lm_head (multimodal wrappers like Gemma3 otherwise return no logits)
        verify_model = lm_model
        if info.lm_head is not None and self.model is not lm_model \
                and not hasattr(lm_model, "lm_head"):
            verify_model = self.model
        verifier = EquivalenceVerifier(verify_model, info)
        ref_logits = None

        if self.do_verify:
            if self.verbose:
                self._print_header("Stage 2: Pre-Transform Snapshot")

            if self.tokenizer is not None:
                if self.verbose:
                    print("  Computing reference logits...")
                ref_logits = verifier.snapshot_output(self.tokenizer)
                if self.verbose:
                    if ref_logits is not None:
                        print(f"  Reference logits shape: {ref_logits.shape}")
                    else:
                        print("  Reference logits unavailable (model output has no logits — verification skipped)")
            else:
                if self.verbose:
                    print("  No tokenizer provided, skipping output verification.")

        # ====== Stage 3: Apply Transforms ======
        if self.verbose:
            self._print_header("Stage 3: Hierarchical Symmetry Transforms")

        transforms = SymmetryTransforms(info, scale_range=self.scale_range)

        # Phase 1: Global Boundary
        if self.verbose:
            print("  Phase 1: Global Residual Stream Permutation (Lemma 1)...")
        transforms.apply_lemma1_residual_permutation()

        if self.verbose:
            print("  Phase 1: Final Norm Scaling (Lemma 3)...")
        transforms.apply_lemma3_final_norm_scaling()

        # Phase 2: Layer-Internal
        num_layers = len(info.layer_components)
        for layer_idx, lc in enumerate(info.layer_components):
            if self.verbose and (layer_idx % max(1, num_layers // 4) == 0 or layer_idx == num_layers - 1):
                extra = ""
                if lc.moe_experts is not None:
                    extra = ", MoE (Lemma 12)"
                print(f"  Phase 2: Layer {layer_idx}/{num_layers-1} "
                      f"[Lemmas 5-10: NormScale, V-O Scale, QK NormScale, "
                      f"HeadPerm, MLP Perm, UpDown Scale{extra}]...")

            with torch.no_grad():
                # Resolve fused projections into virtual separate tensors
                has_fused_attn = transforms._resolve_fused_attn(lc)
                has_fused_mlp = transforms._resolve_fused_mlp(lc)

                # Boundary A: Norm -> Linear scaling (Lemmas 5 & 8)
                transforms.apply_lemma5_8_norm_scaling(lc)

                # Boundary B: Attention internal (Lemmas 6 & 7)
                transforms.apply_lemma7_vo_scaling(lc)
                transforms.apply_qk_norm_scaling(lc)
                transforms.apply_lemma6_head_permutation(lc)

                # Boundary C: MLP internal (Lemmas 9 & 10)
                transforms.apply_lemma9_mlp_permutation(lc)
                transforms.apply_lemma10_updown_scaling(lc)

                # Boundary D: MoE expert-internal transforms (Lemma 12)
                transforms.apply_moe_neuron_permutation(lc)
                transforms.apply_moe_updown_scaling(lc)

                # Write back fused projections
                if has_fused_attn:
                    transforms._writeback_fused_attn(lc)
                if has_fused_mlp:
                    transforms._writeback_fused_mlp(lc)

        transforms.applied_transforms.append("ALL_COMPLETE")

        t_transform = time.time()
        if self.verbose:
            print(f"\n  Transforms completed in {t_transform - t_start:.2f}s")
            print(f"  Applied: {', '.join(transforms.applied_transforms)}")

        # ====== Stage 4: Verification ======
        equiv_result = None
        if self.do_verify:
            if self.verbose:
                self._print_header("Stage 4: Verification")
                print("  Checking functional equivalence...")

            equiv_result = verifier.verify_equivalence(ref_logits, self.tokenizer)

            if self.verbose:
                print_verification_results(equiv_result)

        t_end = time.time()
        if self.verbose:
            self._print_header("Defense Complete")
            print(f"  Total time: {t_end - t_start:.2f}s")

        self.last_report = {
            "transform_time_s": t_transform - t_start,
            "total_time_s": t_end - t_start,
            "applied_transforms": list(transforms.applied_transforms),
        }
        if equiv_result is not None:
            self.last_report["equivalence"] = equiv_result

        return self.model

    def _maybe_untie_lm_head(self, info: ArchitectureInfo):
        """Clone lm_head off embed_tokens for tied models.

        With tied weights, scaling gamma_final would require inversely
        scaling lm_head columns, which would also rescale the (shared)
        embedding rows and break equivalence. Cloning the tensor unties the
        two roles and lets the scaling lemmas apply unchanged.
        """
        if not info.tie_word_embeddings:
            return
        if info.lm_head is None or info.embed_tokens is None:
            return
        if info.lm_head.weight.data_ptr() != info.embed_tokens.weight.data_ptr():
            return
        with torch.no_grad():
            new_w = info.embed_tokens.weight.data.clone()
        info.lm_head.weight = nn.Parameter(
            new_w, requires_grad=info.embed_tokens.weight.requires_grad)
        info.tie_word_embeddings = False
        wrapper_cfg = getattr(self.model, "config", None)
        for cfg in (wrapper_cfg, getattr(wrapper_cfg, "text_config", None)):
            if cfg is not None:
                try:
                    cfg.tie_word_embeddings = False
                except Exception:
                    pass
        if self.verbose:
            print("  Untied lm_head from embed_tokens")

    def _validate_architecture(self, info: ArchitectureInfo):
        """Check that we found essential components."""
        issues = []
        if info.embed_tokens is None:
            issues.append("embed_tokens not found")
        if info.final_norm is None:
            issues.append("final_norm not found")
        if info.lm_head is None:
            issues.append("lm_head not found")
        if not info.layer_components:
            issues.append("no transformer layers found")
        if info.hidden_size == 0:
            issues.append("hidden_size = 0")

        if info.layer_components:
            lc = info.layer_components[0]
            if lc.q_proj is None:
                issues.append("q_proj not found in layer 0")
            if lc.v_proj is None:
                issues.append("v_proj not found in layer 0")
            if lc.gate_proj is None and lc.up_proj is None:
                issues.append("MLP projections not found in layer 0")

        if issues:
            msg = "Architecture validation warnings:\n" + "\n".join(f"  - {i}" for i in issues)
            if self.verbose:
                print(f"\n  WARNING: {msg}")

    def _print_header(self, title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
