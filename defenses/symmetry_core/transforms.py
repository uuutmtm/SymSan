"""Symmetry transforms: the paper's permutation and scaling operations.
Function names carry the paper's lemma numbers."""

import torch
import torch.nn as nn
from .architecture import ArchitectureInfo, LayerComponents


def _scale_norm_weight(norm: nn.Module, s: torch.Tensor) -> None:
    """Scale a norm module so its EFFECTIVE gamma is multiplied by s in-place.

    Two RMSNorm conventions exist:
      - Standard (Llama, Mistral, Qwen, ...): forward uses `weight * x_norm`.
        Effective gamma = weight                  → weight *= s
      - Gemma family (Gemma, Gemma2, Gemma3):    forward uses `(1 + weight) * x_norm`
        with weight initialized to 0.
        Effective gamma = 1 + weight              → weight = s*(1 + weight) - 1

    Without this branching, lemma2/3/5 silently break on Gemma — see the
    scaling-ablation probe results where logit_diff explodes from 0 to 3+
    as r grows, even though the math is supposed to be exact.
    """
    cls = type(norm).__name__
    is_gemma = "Gemma" in cls
    if is_gemma:
        norm.weight.data = s * (1.0 + norm.weight.data) - 1.0
    else:
        norm.weight.data *= s


class SymmetryTransforms:
    """
    Implements the symmetry transforms of the paper, organized by boundary.
    """

    def __init__(self, info: ArchitectureInfo, scale_range: float = 0.05):
        self.info = info
        self.scale_range = scale_range
        self.applied_transforms = []

    # ========================================================================
    # Fused Projection Handling
    # ========================================================================

    def _resolve_fused_attn(self, lc: LayerComponents):
        """
        If qkv_proj is fused, split its weight into virtual q/k/v weight tensors.
        Returns True if fused and needs writeback after transforms.
        """
        if lc.qkv_proj is None or lc.q_proj is not None:
            return False  # Not fused, or already separate

        info = self.info
        w = lc.qkv_proj.weight.data  # shape: [(H*d + Hkv*d + Hkv*d), hidden]
        q_size = info.num_heads * info.head_dim
        k_size = info.num_kv_heads * info.head_dim
        v_size = info.num_kv_heads * info.head_dim

        # Store original reference and split
        lc._fused_qkv_ref = lc.qkv_proj
        q_w, k_w, v_w = w.split([q_size, k_size, v_size], dim=0)

        # Create lightweight proxy objects that hold weight.data
        lc.q_proj = _WeightProxy(q_w)
        lc.k_proj = _WeightProxy(k_w)
        lc.v_proj = _WeightProxy(v_w)

        # Handle bias if present
        if hasattr(lc.qkv_proj, "bias") and lc.qkv_proj.bias is not None:
            b = lc.qkv_proj.bias.data
            q_b, k_b, v_b = b.split([q_size, k_size, v_size], dim=0)
            lc.q_proj.bias = q_b
            lc.k_proj.bias = k_b
            lc.v_proj.bias = v_b
            lc._fused_qkv_has_bias = True
        else:
            lc._fused_qkv_has_bias = False

        return True

    def _writeback_fused_attn(self, lc: LayerComponents):
        """Write split q/k/v back into fused qkv_proj."""
        if not hasattr(lc, "_fused_qkv_ref"):
            return

        lc._fused_qkv_ref.weight.data = torch.cat([
            lc.q_proj.weight.data, lc.k_proj.weight.data, lc.v_proj.weight.data
        ], dim=0)

        if lc._fused_qkv_has_bias:
            lc._fused_qkv_ref.bias.data = torch.cat([
                lc.q_proj.bias, lc.k_proj.bias, lc.v_proj.bias
            ], dim=0)

        # Clean up proxies
        lc.q_proj = None
        lc.k_proj = None
        lc.v_proj = None
        del lc._fused_qkv_ref
        del lc._fused_qkv_has_bias

    def _resolve_fused_mlp(self, lc: LayerComponents):
        """
        If gate_up_proj is fused, split into virtual gate/up weight tensors.
        Returns True if fused and needs writeback.
        """
        if lc.gate_up_proj is None or lc.gate_proj is not None:
            return False

        info = self.info
        w = lc.gate_up_proj.weight.data  # shape: [2 * intermediate, hidden]
        half = info.intermediate_size

        lc._fused_gate_up_ref = lc.gate_up_proj
        gate_w, up_w = w.split([half, half], dim=0)
        lc.gate_proj = _WeightProxy(gate_w)
        lc.up_proj = _WeightProxy(up_w)

        if hasattr(lc.gate_up_proj, "bias") and lc.gate_up_proj.bias is not None:
            b = lc.gate_up_proj.bias.data
            gate_b, up_b = b.split([half, half], dim=0)
            lc.gate_proj.bias = gate_b
            lc.up_proj.bias = up_b
            lc._fused_gate_up_has_bias = True
        else:
            lc._fused_gate_up_has_bias = False

        return True

    def _writeback_fused_mlp(self, lc: LayerComponents):
        """Write split gate/up back into fused gate_up_proj."""
        if not hasattr(lc, "_fused_gate_up_ref"):
            return

        lc._fused_gate_up_ref.weight.data = torch.cat([
            lc.gate_proj.weight.data, lc.up_proj.weight.data
        ], dim=0)

        if lc._fused_gate_up_has_bias:
            lc._fused_gate_up_ref.bias.data = torch.cat([
                lc.gate_proj.bias, lc.up_proj.bias
            ], dim=0)

        lc.gate_proj = None
        lc.up_proj = None
        del lc._fused_gate_up_ref
        del lc._fused_gate_up_has_bias

    # ========================================================================
    # Level 1: Global Boundary Transforms
    # ========================================================================

    def apply_lemma1_residual_permutation(self):
        """
        Lemma 1: Residual Stream Permutation (P1).

        Proof sketch:
          RMS(x[P]) = RMS(x) because permutation preserves statistics.
          RMSNorm(x[P], g[P]) = RMSNorm(x, g)[P]
          Signal propagates consistently from embed to lm_head.
        """
        info = self.info
        h = info.hidden_size
        device, dtype = self._get_device_dtype()

        P = torch.randperm(h, device=device)

        # Global components
        if info.embed_tokens is not None:
            w = info.embed_tokens.weight.data
            info.embed_tokens.weight.data = w[:, P]

        if info.lm_head is not None and not info.tie_word_embeddings:
            w = info.lm_head.weight.data
            info.lm_head.weight.data = w[:, P]

        if info.final_norm is not None:
            info.final_norm.weight.data = info.final_norm.weight.data[P]
            if hasattr(info.final_norm, "bias") and info.final_norm.bias is not None:
                info.final_norm.bias.data = info.final_norm.bias.data[P]

        # Multimodal projector: its output feeds the language-model residual
        # stream, so its output side must follow the same permutation.
        # Linear-style projector (y = Wx): permute output rows.
        if info.mm_projector is not None:
            w = info.mm_projector.weight.data
            info.mm_projector.weight.data = w[P, :]
            if hasattr(info.mm_projector, "bias") and info.mm_projector.bias is not None:
                info.mm_projector.bias.data = info.mm_projector.bias.data[P]
        # Matmul-style projector (y = x @ W, e.g. Gemma3): permute output cols.
        if info.mm_projection_param is not None:
            w = info.mm_projection_param.data
            info.mm_projection_param.data = w[:, P]

        # Per-layer components
        for lc in info.layer_components:
            # All norms: weight permuted
            for norm in [lc.input_layernorm, lc.post_attention_layernorm,
                         lc.pre_feedforward_layernorm, lc.post_feedforward_layernorm]:
                if norm is not None:
                    norm.weight.data = norm.weight.data[P]
                    if hasattr(norm, "bias") and norm.bias is not None:
                        norm.bias.data = norm.bias.data[P]

            # Input projections: column permutation
            for proj in [lc.q_proj, lc.k_proj, lc.v_proj,
                         lc.gate_proj, lc.up_proj, lc.qkv_proj, lc.gate_up_proj]:
                if proj is not None:
                    proj.weight.data = proj.weight.data[:, P]

            # Output projections: row permutation
            for proj in [lc.o_proj, lc.down_proj]:
                if proj is not None:
                    proj.weight.data = proj.weight.data[P, :]
                    if hasattr(proj, "bias") and proj.bias is not None:
                        proj.bias.data = proj.bias.data[P]

            # MoE expert tensors: apply residual-stream permutation to input/output dims
            if lc.moe_experts is not None:
                exp = lc.moe_experts
                exp.gate_up_proj.data = exp.gate_up_proj.data[:, :, P]
                exp.down_proj.data = exp.down_proj.data[:, P, :]
            if lc.moe_router is not None and hasattr(lc.moe_router, "weight"):
                lc.moe_router.weight.data = lc.moe_router.weight.data[:, P]

        self.applied_transforms.append("Lemma1_ResidualPerm")

    def apply_lemma3_final_norm_scaling(self):
        """
        Lemma 3 (paper): I-O scaling pair, C1.

        Proof:
          RMSNorm(x, s.g) = s . RMSNorm(x, g)
          (W/s) . (s . RMSNorm(x,g)) = W . RMSNorm(x,g)

        Constraint: Skipped when tie_word_embeddings=True.
        """
        info = self.info

        if info.tie_word_embeddings:
            self.applied_transforms.append("Lemma3_FinalNormScaling(SKIPPED:tied)")
            return

        if info.final_norm is None or info.lm_head is None:
            return

        device = info.final_norm.weight.device
        dtype = info.final_norm.weight.dtype
        h = info.hidden_size
        s = self._random_scale(h, device, dtype)

        _scale_norm_weight(info.final_norm, s)
        if hasattr(info.final_norm, "bias") and info.final_norm.bias is not None:
            info.final_norm.bias.data *= s
        info.lm_head.weight.data /= s.view(1, -1)

        self.applied_transforms.append("Lemma3_FinalNormScaling")

    # ========================================================================
    # Level 2: Layer-Internal Transforms
    # ========================================================================

    def apply_lemma5_8_norm_scaling(self, lc: LayerComponents):
        """
        Norm-to-projection scaling: paper Lemma 5 (C2, input norm -> Q/K/V)
        and paper Lemma 8 (C4, post-attention norm -> gate/up). The same
        mechanism serves both boundaries; for MoE layers it also compensates
        the expert router (paper Lemma 12).

        Proof:
          RMSNorm(x, s.g) = s . RMSNorm(x, g)
          (W/s) . (s . RMSNorm(x,g)) = W . RMSNorm(x,g)
        """
        info = self.info
        h = info.hidden_size

        # Boundary A-1: input_layernorm -> Q/K/V
        if lc.input_layernorm is not None:
            device = lc.input_layernorm.weight.device
            dtype = lc.input_layernorm.weight.dtype
            s_in = self._random_scale(h, device, dtype)

            _scale_norm_weight(lc.input_layernorm, s_in)
            if hasattr(lc.input_layernorm, "bias") and lc.input_layernorm.bias is not None:
                lc.input_layernorm.bias.data *= s_in

            # Scale separate or fused attention input projections
            for proj in [lc.q_proj, lc.k_proj, lc.v_proj]:
                if proj is not None:
                    proj.weight.data /= s_in.view(1, -1)
            if lc.qkv_proj is not None and lc.q_proj is None:
                lc.qkv_proj.weight.data /= s_in.view(1, -1)

        # Boundary A-2: post_attention_layernorm -> Gate/Up
        # (use pre_feedforward_layernorm if it exists, e.g. Gemma3)
        post_norm = lc.pre_feedforward_layernorm or lc.post_attention_layernorm
        if post_norm is not None:
            device = post_norm.weight.device
            dtype = post_norm.weight.dtype
            s_post = self._random_scale(h, device, dtype)

            _scale_norm_weight(post_norm, s_post)
            if hasattr(post_norm, "bias") and post_norm.bias is not None:
                post_norm.bias.data *= s_post

            for proj in [lc.gate_proj, lc.up_proj]:
                if proj is not None:
                    proj.weight.data /= s_post.view(1, -1)
            if lc.gate_up_proj is not None and lc.gate_proj is None:
                lc.gate_up_proj.weight.data /= s_post.view(1, -1)
            # MoE: scale expert input weights (gate_up_proj[:, :, H] dim)
            if lc.moe_experts is not None:
                lc.moe_experts.gate_up_proj.data /= s_post.view(1, 1, -1)
            # MoE router also takes residual stream as input
            if lc.moe_router is not None and hasattr(lc.moe_router, "weight"):
                lc.moe_router.weight.data /= s_post.view(1, -1)

    def apply_lemma7_vo_scaling(self, lc: LayerComponents):
        """
        Lemma 7 (paper): Value-Output scaling pair, C3.

        Proof:
          Z_V' = C3 . Z_V  (per head, V output scaled)
          O_out' = softmax . (C3 . Z_V) . (W_O / C3)^T
                 = softmax . Z_V . W_O^T = original  (C3 cancels)
        """
        info = self.info
        if lc.v_proj is None or lc.o_proj is None:
            return

        num_kv_heads = info.num_kv_heads
        num_groups = info.num_groups
        head_dim = info.head_dim

        device = lc.v_proj.weight.device
        dtype = lc.v_proj.weight.dtype

        v_w = lc.v_proj.weight.data.view(num_kv_heads, head_dim, -1)
        o_w = lc.o_proj.weight.data.view(-1, info.num_heads, head_dim)

        for kv_idx in range(num_kv_heads):
            c3 = self._random_scale(head_dim, device, dtype)
            v_w[kv_idx] *= c3.view(-1, 1)
            for q_idx in range(kv_idx * num_groups, (kv_idx + 1) * num_groups):
                o_w[:, q_idx, :] /= c3.view(1, -1)

        lc.v_proj.weight.data = v_w.reshape(-1, v_w.shape[-1])
        lc.o_proj.weight.data = o_w.reshape(o_w.shape[0], -1)

    def apply_qk_norm_scaling(self, lc: LayerComponents):
        """
        QK-norm scaling — architecture-specific instance of the norm-to-linear
        scaling pair (paper Lemma 5 mechanism) for models with per-head q/k
        RMSNorm (Gemma-3, Qwen3, OLMoE). Not separately numbered in the paper.

        Scales q_norm by D and k_norm by D^{-1}, where
        D = diag(c_1..c_{dh/2}, c_1..c_{dh/2}) is constant on each RoPE pair
        (i, i + dh/2) used by rotate_half-style RoPE, and therefore commutes
        with the 2x2 rotation blocks.

        Proof:
          q' = RMSNorm(q_raw, D.g_q) = D . RMSNorm(q_raw, g_q)
          k' = RMSNorm(k_raw, D^{-1}.g_k) = D^{-1} . RMSNorm(k_raw, g_k)
          score = q'^T . k' = (D.q)^T . (D^{-1}.k) = q^T . k  (D cancels)
        """
        info = self.info
        if lc.q_norm is None or not hasattr(lc.q_norm, "weight"):
            return
        if lc.k_norm is None or not hasattr(lc.k_norm, "weight"):
            return

        num_kv_heads = info.num_kv_heads
        num_groups = info.num_groups
        head_dim = info.head_dim
        device = lc.q_norm.weight.device
        dtype = lc.q_norm.weight.dtype

        # Detect Gemma-style (1+w) convention; convert to "effective gamma" view,
        # apply scaling, then convert back. This keeps the per-head reshaping logic
        # identical for both norm conventions.
        q_is_gemma = "Gemma" in type(lc.q_norm).__name__
        k_is_gemma = "Gemma" in type(lc.k_norm).__name__

        qn_eff = (1.0 + lc.q_norm.weight.data) if q_is_gemma else lc.q_norm.weight.data.clone()
        kn_eff = (1.0 + lc.k_norm.weight.data) if k_is_gemma else lc.k_norm.weight.data.clone()

        if qn_eff.shape[0] == head_dim:
            pair_scales = self._random_scale(head_dim // 2, device, dtype)
            # Constant on each RoPE pair (i, i + head_dim/2) as used by
            # rotate_half-style RoPE (HF Gemma3/Qwen3/OLMoE), so that D
            # commutes with the 2x2 rotation blocks exactly.
            D = pair_scales.repeat(2)
            qn_eff *= D
            kn_eff /= D
        elif qn_eff.shape[0] == info.num_heads * head_dim:
            qn_reshaped = qn_eff.view(info.num_heads, head_dim)
            kn_reshaped = kn_eff.view(num_kv_heads, head_dim)
            for kv_idx in range(num_kv_heads):
                pair_scales = self._random_scale(head_dim // 2, device, dtype)
                # Constant on each RoPE pair (i, i + head_dim/2) - see above.
                D = pair_scales.repeat(2)
                kn_reshaped[kv_idx] /= D
                for q_idx in range(kv_idx * num_groups, (kv_idx + 1) * num_groups):
                    qn_reshaped[q_idx] *= D
            qn_eff = qn_reshaped.view(-1)
            kn_eff = kn_reshaped.view(-1)
        else:
            return

        lc.q_norm.weight.data = (qn_eff - 1.0) if q_is_gemma else qn_eff
        lc.k_norm.weight.data = (kn_eff - 1.0) if k_is_gemma else kn_eff

    def apply_lemma6_head_permutation(self, lc: LayerComponents):
        """
        Lemma 6 (paper): Head permutation pair, P2 (RoPE-compatible).

        Proof: Multi-head output = sum_h head_h . W_O_h. Sum is commutative.
        For GQA: permute KV-head groups as a whole.
        RoPE-compatible because each head uses the same set of frequencies.
        """
        info = self.info
        if lc.q_proj is None or lc.o_proj is None:
            return

        num_kv_heads = info.num_kv_heads
        num_groups = info.num_groups
        head_dim = info.head_dim
        device = lc.q_proj.weight.device

        kv_perm = torch.randperm(num_kv_heads, device=device)

        q_w = lc.q_proj.weight.data.view(num_kv_heads, num_groups, head_dim, -1)
        lc.q_proj.weight.data = q_w[kv_perm].reshape(-1, q_w.shape[-1])

        k_w = lc.k_proj.weight.data.view(num_kv_heads, head_dim, -1)
        lc.k_proj.weight.data = k_w[kv_perm].reshape(-1, k_w.shape[-1])

        v_w = lc.v_proj.weight.data.view(num_kv_heads, head_dim, -1)
        lc.v_proj.weight.data = v_w[kv_perm].reshape(-1, v_w.shape[-1])

        o_w = lc.o_proj.weight.data.view(-1, num_kv_heads, num_groups, head_dim)
        lc.o_proj.weight.data = o_w[:, kv_perm].reshape(o_w.shape[0], -1)

        if lc.q_norm is not None and hasattr(lc.q_norm, "weight"):
            qn = lc.q_norm.weight.data
            if qn.shape[0] == info.num_heads * head_dim:
                qn = qn.view(num_kv_heads, num_groups, head_dim)
                lc.q_norm.weight.data = qn[kv_perm].reshape(-1)

        if lc.k_norm is not None and hasattr(lc.k_norm, "weight"):
            kn = lc.k_norm.weight.data
            if kn.shape[0] == num_kv_heads * head_dim:
                kn = kn.view(num_kv_heads, head_dim)
                lc.k_norm.weight.data = kn[kv_perm].reshape(-1)

    def apply_lemma9_mlp_permutation(self, lc: LayerComponents):
        """
        Lemma 9 (paper): FFN neuron permutation pair, P3.

        Proof: SiLU/GELU are pointwise, element-wise product is permutation-equivariant.
               down_proj column permutation cancels the reordering.
        """
        if lc.gate_proj is None or lc.down_proj is None:
            return

        d_ff = lc.gate_proj.weight.shape[0]
        device = lc.gate_proj.weight.device
        pi = torch.randperm(d_ff, device=device)

        lc.gate_proj.weight.data = lc.gate_proj.weight.data[pi]
        if lc.up_proj is not None:
            lc.up_proj.weight.data = lc.up_proj.weight.data[pi]

        lc.down_proj.weight.data = lc.down_proj.weight.data[:, pi]

        # Biases
        if hasattr(lc.gate_proj, "bias") and lc.gate_proj.bias is not None:
            lc.gate_proj.bias.data = lc.gate_proj.bias.data[pi]
        if lc.up_proj is not None and hasattr(lc.up_proj, "bias") and lc.up_proj.bias is not None:
            lc.up_proj.bias.data = lc.up_proj.bias.data[pi]

    def apply_lemma10_updown_scaling(self, lc: LayerComponents):
        """
        Lemma 10 (paper): Up-down scaling pair, C5.

        Proof:
          h = Act(gate(x)) . up(x)
          h' = Act(gate(x)) . (c . up(x)) = c . h
          down'(h') = (down/c) . (c . h) = down . h

        Note: Cannot scale gate_proj because SiLU/GELU are NOT homogeneous.
        """
        if lc.up_proj is None or lc.down_proj is None:
            return

        d_ff = lc.up_proj.weight.shape[0]
        device = lc.up_proj.weight.device
        dtype = lc.up_proj.weight.dtype
        c = self._random_scale(d_ff, device, dtype)

        lc.up_proj.weight.data *= c.view(-1, 1)
        if hasattr(lc.up_proj, "bias") and lc.up_proj.bias is not None:
            lc.up_proj.bias.data *= c

        lc.down_proj.weight.data /= c.view(1, -1)

    # ========================================================================
    # MoE Transforms (paper Lemma 12)
    # ========================================================================

    def apply_moe_neuron_permutation(self, lc: LayerComponents):
        """
        MoE expert-internal neuron permutation (part of paper Lemma 12).

        Proof (per expert e, independently):
          gate_up_proj[e]: [2*D_ff, H] — first D_ff rows=gate, last D_ff rows=up
          For expert e with permutation π_e on D_ff dimension:
            gate'[e][π_e, :] = gate[e]   →  act(gate'[e])[π_e] = act(gate[e])[π_e]
            up'[e][π_e, :]   = up[e]     →  up'[e][π_e] = up[e][π_e]
            element-wise product: act(gate')[π_e] * up'[π_e] = (act(gate) * up)[π_e]
            down'[e][:, π_e] = down[e]   →  down'[e] @ (act*up)[π_e]
                             = down[e] @ (act*up)  ✓
        """
        if lc.moe_experts is None:
            return

        E = lc.num_experts
        device = lc.moe_experts.gate_up_proj.device
        # gate_up_proj: [E, 2*D_ff, H]; down_proj: [E, H, D_ff]
        D_ff = lc.moe_experts.down_proj.shape[2]

        for e in range(E):
            pi_e = torch.randperm(D_ff, device=device)
            # Permute gate rows (first D_ff) and up rows (last D_ff) independently
            lc.moe_experts.gate_up_proj.data[e, :D_ff, :] = \
                lc.moe_experts.gate_up_proj.data[e, :D_ff, :][pi_e]
            lc.moe_experts.gate_up_proj.data[e, D_ff:, :] = \
                lc.moe_experts.gate_up_proj.data[e, D_ff:, :][pi_e]
            # Permute down_proj columns
            lc.moe_experts.down_proj.data[e, :, :] = \
                lc.moe_experts.down_proj.data[e, :, pi_e]

    def apply_moe_updown_scaling(self, lc: LayerComponents):
        """
        MoE expert-internal up-down scaling (part of paper Lemma 12).

        Proof (per expert e, independently):
          h = act(gate(x)) ⊙ up(x)
          h' = act(gate(x)) ⊙ (c_e ⊙ up(x)) = c_e ⊙ h
          down'(h') = (down / c_e) @ (c_e ⊙ h) = down @ h  ✓

          gate_proj cannot be scaled because SiLU/GELU are not homogeneous.
          Only up (last D_ff rows of gate_up_proj) and down_proj are scaled.
        """
        if lc.moe_experts is None:
            return

        E = lc.num_experts
        device = lc.moe_experts.gate_up_proj.device
        dtype = lc.moe_experts.gate_up_proj.dtype
        D_ff = lc.moe_experts.down_proj.shape[2]

        for e in range(E):
            c_e = self._random_scale(D_ff, device, dtype)
            # Scale up part of gate_up_proj (last D_ff rows): [D_ff, H] * c_e.view(-1,1)
            lc.moe_experts.gate_up_proj.data[e, D_ff:, :] *= c_e.view(-1, 1)
            # Inverse-scale down_proj columns: [H, D_ff] / c_e.view(1,-1)
            lc.moe_experts.down_proj.data[e, :, :] /= c_e.view(1, -1)

    def apply_all(self):
        """Apply the complete transformation pipeline in correct order."""
        info = self.info

        with torch.no_grad():
            # Phase 1: Global Boundary
            self.apply_lemma1_residual_permutation()
            self.apply_lemma3_final_norm_scaling()

            # Phase 2: Layer-Internal Boundaries (per layer)
            for layer_idx, lc in enumerate(info.layer_components):
                # Resolve fused projections into virtual separate tensors
                has_fused_attn = self._resolve_fused_attn(lc)
                has_fused_mlp = self._resolve_fused_mlp(lc)

                # Boundary A: Norm -> Linear scaling (Lemmas 5 & 8)
                self.apply_lemma5_8_norm_scaling(lc)

                # Boundary B: Attention internal (Lemmas 6 & 7)
                self.apply_lemma7_vo_scaling(lc)
                self.apply_qk_norm_scaling(lc)
                self.apply_lemma6_head_permutation(lc)

                # Boundary C: MLP internal (Lemmas 9 & 10)
                self.apply_lemma9_mlp_permutation(lc)
                self.apply_lemma10_updown_scaling(lc)

                # Boundary D: MoE expert-internal transforms (Lemma 12)
                self.apply_moe_neuron_permutation(lc)
                self.apply_moe_updown_scaling(lc)

                # Write back fused projections
                if has_fused_attn:
                    self._writeback_fused_attn(lc)
                if has_fused_mlp:
                    self._writeback_fused_mlp(lc)

        self.applied_transforms.append("ALL_COMPLETE")

    # ========================================================================
    # Utilities
    # ========================================================================

    def _random_scale(self, size: int, device, dtype) -> torch.Tensor:
        """Generate a random positive scaling vector with values in [1-r, 1+r]."""
        r = self.scale_range
        return torch.rand(size, device=device, dtype=dtype) * 2 * r + (1 - r)

    def _get_device_dtype(self):
        """Get device and dtype from the first available parameter."""
        info = self.info
        if info.embed_tokens is not None:
            p = info.embed_tokens.weight
            return p.device, p.dtype
        if info.layer_components:
            lc = info.layer_components[0]
            if lc.input_layernorm is not None:
                return lc.input_layernorm.weight.device, lc.input_layernorm.weight.dtype
        for p in self.info.layers[0].parameters():
            return p.device, p.dtype
        return torch.device("cpu"), torch.float16


class _WeightProxy:
    """
    Lightweight proxy that mimics nn.Linear for transform operations.
    Holds a weight tensor (view into fused weight) and optional bias.
    """
    def __init__(self, weight_data: torch.Tensor):
        self.weight = _DataHolder(weight_data)
        self.bias = None

    def __bool__(self):
        return True


class _DataHolder:
    """Mimics parameter.data access pattern."""
    def __init__(self, data: torch.Tensor):
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype
