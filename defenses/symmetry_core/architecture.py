"""Detects the model structure the symmetry transforms operate on."""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerComponents:
    """Components identified within a single transformer layer."""
    # Normalization layers
    input_layernorm: Optional[nn.Module] = None
    post_attention_layernorm: Optional[nn.Module] = None
    pre_feedforward_layernorm: Optional[nn.Module] = None   # Gemma3-style
    post_feedforward_layernorm: Optional[nn.Module] = None  # Gemma3-style

    # Attention projections (separate)
    q_proj: Optional[nn.Linear] = None
    k_proj: Optional[nn.Linear] = None
    v_proj: Optional[nn.Linear] = None
    o_proj: Optional[nn.Linear] = None

    # Attention projections (fused) — e.g. Phi-4
    qkv_proj: Optional[nn.Linear] = None  # fused Q+K+V

    # QK normalization (Qwen3 / Gemma3-style)
    q_norm: Optional[nn.Module] = None
    k_norm: Optional[nn.Module] = None

    # MLP projections (separate)
    gate_proj: Optional[nn.Linear] = None
    up_proj: Optional[nn.Linear] = None
    down_proj: Optional[nn.Linear] = None

    # MLP projections (fused) — e.g. Phi-4
    gate_up_proj: Optional[nn.Linear] = None  # fused gate+up

    # MoE components — e.g. OLMoE
    moe_experts: Optional[nn.Module] = None   # OlmoeExperts / similar
    moe_router: Optional[nn.Module] = None    # OlmoeTopKRouter / similar
    num_experts: int = 0

    # Module references
    self_attn: Optional[nn.Module] = None
    mlp: Optional[nn.Module] = None


@dataclass
class ArchitectureInfo:
    """Complete architecture description extracted from model inspection."""
    # Structural parameters
    hidden_size: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_groups: int = 1  # num_heads // num_kv_heads (for GQA)
    intermediate_size: int = 0
    num_layers: int = 0
    vocab_size: int = 0

    # Special architecture flags
    tie_word_embeddings: bool = False
    has_qk_norm: bool = False
    has_post_feedforward_layernorm: bool = False
    has_fused_qkv: bool = False
    has_fused_gate_up: bool = False
    norm_has_bias: bool = False
    linear_has_bias: bool = False
    is_moe: bool = False
    num_experts: int = 0

    # Global component references
    embed_tokens: Optional[nn.Module] = None
    lm_head: Optional[nn.Module] = None
    final_norm: Optional[nn.Module] = None
    mm_projector: Optional[nn.Module] = None
    mm_projection_param: Optional[torch.Tensor] = None
    layers: list = field(default_factory=list)

    # Per-layer component references
    layer_components: list = field(default_factory=list)

    # Detected model family (informational only)
    model_family: str = "unknown"



# --- Name pattern tables for architecture-agnostic module discovery ---

_BASE_MODEL_ATTRS = ["model", "transformer", "gpt_neox", "backbone"]
_LM_WRAPPER_ATTRS = ["language_model"]  # Gemma3 conditional generation wrapper

_EMBED_ATTRS = ["embed_tokens", "wte", "word_embeddings", "embed_in"]
_LAYERS_ATTRS = ["layers", "h", "blocks", "layer"]
_FINAL_NORM_ATTRS = ["norm", "ln_f", "final_layer_norm", "final_layernorm", "norm_f"]
_LM_HEAD_ATTRS = ["lm_head", "output", "embed_out"]

_INPUT_NORM_ATTRS = [
    "input_layernorm", "ln_1", "attention_norm",
    "norm1", "pre_attention_layernorm",
]
_POST_ATTN_NORM_ATTRS = [
    "post_attention_layernorm", "ln_2", "ffn_norm",
    "norm2",
]
_PRE_FFW_NORM_ATTRS = ["pre_feedforward_layernorm"]
_POST_FFW_NORM_ATTRS = ["post_feedforward_layernorm"]
_SELF_ATTN_ATTRS = ["self_attn", "attention", "attn", "self_attention"]
_MLP_ATTRS = ["mlp", "feed_forward", "ffn", "ff"]

# Separate attention projections
_Q_PROJ_ATTRS = ["q_proj", "query", "q"]
_K_PROJ_ATTRS = ["k_proj", "key", "k"]
_V_PROJ_ATTRS = ["v_proj", "value", "v"]
_O_PROJ_ATTRS = ["o_proj", "dense", "out_proj"]
_QKV_FUSED_ATTRS = ["qkv_proj", "query_key_value", "c_attn"]
_QK_NORM_ATTRS = ["q_norm", "q_layernorm"]
_KK_NORM_ATTRS = ["k_norm", "k_layernorm"]

# Separate MLP projections
_GATE_PROJ_ATTRS = ["gate_proj", "w1", "gate"]
_UP_PROJ_ATTRS = ["up_proj", "w3", "up", "fc1"]
_DOWN_PROJ_ATTRS = ["down_proj", "w2", "down", "fc2"]
_GATE_UP_FUSED_ATTRS = ["gate_up_proj"]


def _find_attr(module: nn.Module, candidates: list) -> Optional[nn.Module]:
    """Try to find a sub-module by checking a list of candidate attribute names."""
    for attr_name in candidates:
        if hasattr(module, attr_name):
            sub = getattr(module, attr_name)
            if sub is not None:
                return sub
    return None


def _find_attr_name(module: nn.Module, candidates: list) -> Optional[str]:
    """Like _find_attr but returns the attribute name instead of the module."""
    for attr_name in candidates:
        if hasattr(module, attr_name):
            sub = getattr(module, attr_name)
            if sub is not None:
                return attr_name
    return None


class ArchitectureAnalyzer:
    """
    Automatically identifies the architecture of any HuggingFace CausalLM.

    Uses pattern-based attribute discovery to locate modules generically.
    Handles: Llama, Mistral, Qwen, Gemma3, Phi-4, DeepSeek, and more.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        # Handle multi-modal wrappers (e.g. Gemma3ForConditionalGeneration)
        self.lm_model = self._unwrap_to_lm(model)
        self.config = self.lm_model.config

    def _unwrap_to_lm(self, model: nn.Module) -> nn.Module:
        """Unwrap multi-modal wrappers to find the core CausalLM."""
        # Check direct attributes first (e.g. model.language_model)
        for attr in _LM_WRAPPER_ATTRS:
            if hasattr(model, attr):
                lm = getattr(model, attr)
                if hasattr(lm, "config") and hasattr(lm.config, "hidden_size"):
                    return lm

        # Check one level deeper: model.model.language_model (Gemma3ForConditionalGeneration)
        for base_attr in _BASE_MODEL_ATTRS:
            if hasattr(model, base_attr):
                base = getattr(model, base_attr)
                for attr in _LM_WRAPPER_ATTRS:
                    if hasattr(base, attr):
                        lm = getattr(base, attr)
                        if hasattr(lm, "config") and hasattr(lm.config, "hidden_size"):
                            return lm

        return model

    def analyze(self) -> ArchitectureInfo:
        """Run full architecture analysis. Returns ArchitectureInfo with all references."""
        info = ArchitectureInfo()

        self._extract_config_params(info)
        self._locate_global_components(info)

        if len(info.layers) > 0:
            self._detect_features(info, info.layers[0])

        self._build_layer_components(info)
        info.model_family = self._infer_family()

        return info

    def _extract_config_params(self, info: ArchitectureInfo):
        """Extract structural dimensions from model config."""
        cfg = self.config

        info.hidden_size = getattr(cfg, "hidden_size", 0)
        info.num_heads = getattr(cfg, "num_attention_heads", 0)
        info.num_kv_heads = getattr(cfg, "num_key_value_heads", info.num_heads)
        info.vocab_size = getattr(cfg, "vocab_size", 0)
        info.intermediate_size = getattr(cfg, "intermediate_size", 0)
        info.num_layers = getattr(cfg, "num_hidden_layers", 0)
        info.tie_word_embeddings = getattr(cfg, "tie_word_embeddings", False)

        if info.num_heads > 0 and info.hidden_size > 0:
            info.head_dim = getattr(cfg, "head_dim", info.hidden_size // info.num_heads)
        if info.num_kv_heads > 0:
            info.num_groups = info.num_heads // info.num_kv_heads

    def _locate_global_components(self, info: ArchitectureInfo):
        """Locate embed_tokens, layers, final_norm, lm_head."""
        model = self.lm_model

        # Find base model container
        base = None
        for attr in _BASE_MODEL_ATTRS:
            if hasattr(model, attr):
                base = getattr(model, attr)
                break
        if base is None:
            base = model

        # Embed tokens
        info.embed_tokens = _find_attr(base, _EMBED_ATTRS)
        if info.embed_tokens is None and hasattr(model, "get_input_embeddings"):
            info.embed_tokens = model.get_input_embeddings()

        # Layer container
        layers_module = _find_attr(base, _LAYERS_ATTRS)
        if layers_module is not None:
            info.layers = list(layers_module)
            info.num_layers = len(info.layers)

        # Final normalization
        info.final_norm = _find_attr(base, _FINAL_NORM_ATTRS)

        # LM head
        info.lm_head = _find_attr(model, _LM_HEAD_ATTRS)
        if info.lm_head is None and hasattr(model, "get_output_embeddings"):
            info.lm_head = model.get_output_embeddings()

        # LM head fallback: multimodal wrappers keep lm_head above the text
        # backbone (e.g., Gemma3ForConditionalGeneration.lm_head)
        if info.lm_head is None:
            for holder in (self.model, getattr(self.model, "model", None)):
                if holder is None:
                    continue
                cand = _find_attr(holder, _LM_HEAD_ATTRS)
                if cand is not None and hasattr(cand, "weight"):
                    info.lm_head = cand
                    break

        # Multimodal projector: its output feeds the LM residual stream
        # (e.g., Gemma3Model.multi_modal_projector). Some families wrap the
        # projection Linear in a container module (Gemma3: .mm_soft_emb), so
        # resolve to the inner 2-D weight holder.
        def _resolve_linear(mod):
            # Linear-style projector (y = Wx): output rows = LM hidden
            if mod is None:
                return None
            w = getattr(mod, "weight", None)
            if w is not None and getattr(w, "dim", lambda: 0)() == 2 and w.shape[0] == info.hidden_size:
                return mod
            for child in mod.children():
                w = getattr(child, "weight", None)
                if w is not None and w.dim() == 2 and w.shape[0] == info.hidden_size:
                    return child
            return None

        for holder in (self.model, getattr(self.model, "model", None)):
            if holder is None:
                continue
            for attr in ("multi_modal_projector", "mm_projector"):
                proj_mod = getattr(holder, attr, None)
                if proj_mod is None:
                    continue
                proj = _resolve_linear(proj_mod)
                if proj is not None:
                    info.mm_projector = proj
                else:
                    # Matmul-style projector (y = x @ W, e.g. Gemma3
                    # mm_input_projection_weight of shape (vision_h, text_h)):
                    # LM-hidden dim is the columns.
                    try:
                        direct = dict(proj_mod.named_parameters(recurse=False))
                    except Exception:
                        direct = {}
                    for pname, p in direct.items():
                        if p.dim() == 2 and p.shape[1] == info.hidden_size:
                            info.mm_projection_param = p
                            break
                if info.mm_projector is not None or info.mm_projection_param is not None:
                    break
            if info.mm_projector is not None or info.mm_projection_param is not None:
                break

    def _detect_features(self, info: ArchitectureInfo, sample_layer: nn.Module):
        """Inspect a single layer to detect special architecture features."""
        info.has_post_feedforward_layernorm = (
            _find_attr(sample_layer, _POST_FFW_NORM_ATTRS) is not None
        )

        attn = _find_attr(sample_layer, _SELF_ATTN_ATTRS)
        if attn is not None:
            info.has_qk_norm = _find_attr(attn, _QK_NORM_ATTRS) is not None
            info.has_fused_qkv = _find_attr(attn, _QKV_FUSED_ATTRS) is not None

            # Check bias on whatever projection we can find
            proj = _find_attr(attn, _Q_PROJ_ATTRS) or _find_attr(attn, _QKV_FUSED_ATTRS)
            if proj is not None and hasattr(proj, "bias"):
                info.linear_has_bias = proj.bias is not None

        mlp = _find_attr(sample_layer, _MLP_ATTRS)
        if mlp is not None:
            info.has_fused_gate_up = _find_attr(mlp, _GATE_UP_FUSED_ATTRS) is not None
            # Detect MoE: OlmoeExperts stores gate_up_proj/down_proj as 3-D tensors
            # under an 'experts' sub-module, plus a 'gate' router sub-module
            if hasattr(mlp, "experts") and hasattr(mlp, "gate"):
                experts = mlp.experts
                if (hasattr(experts, "gate_up_proj")
                        and isinstance(experts.gate_up_proj, torch.Tensor)
                        and experts.gate_up_proj.dim() == 3):
                    info.is_moe = True
                    info.num_experts = experts.gate_up_proj.shape[0]

        norm = _find_attr(sample_layer, _INPUT_NORM_ATTRS)
        if norm is not None and hasattr(norm, "bias"):
            info.norm_has_bias = norm.bias is not None

    def _build_layer_components(self, info: ArchitectureInfo):
        """Build per-layer component reference list."""
        info.layer_components = []
        for layer in info.layers:
            lc = LayerComponents()

            # Norms
            lc.input_layernorm = _find_attr(layer, _INPUT_NORM_ATTRS)
            lc.post_attention_layernorm = _find_attr(layer, _POST_ATTN_NORM_ATTRS)
            lc.pre_feedforward_layernorm = _find_attr(layer, _PRE_FFW_NORM_ATTRS)
            lc.post_feedforward_layernorm = _find_attr(layer, _POST_FFW_NORM_ATTRS)

            # Attention
            lc.self_attn = _find_attr(layer, _SELF_ATTN_ATTRS)
            if lc.self_attn is not None:
                lc.q_proj = _find_attr(lc.self_attn, _Q_PROJ_ATTRS)
                lc.k_proj = _find_attr(lc.self_attn, _K_PROJ_ATTRS)
                lc.v_proj = _find_attr(lc.self_attn, _V_PROJ_ATTRS)
                lc.o_proj = _find_attr(lc.self_attn, _O_PROJ_ATTRS)
                lc.qkv_proj = _find_attr(lc.self_attn, _QKV_FUSED_ATTRS)
                lc.q_norm = _find_attr(lc.self_attn, _QK_NORM_ATTRS)
                lc.k_norm = _find_attr(lc.self_attn, _KK_NORM_ATTRS)

            # MLP
            lc.mlp = _find_attr(layer, _MLP_ATTRS)
            if lc.mlp is not None:
                # MoE: detect OlmoeExperts-style experts sub-module FIRST
                # Must do this before searching for gate_proj, because mlp.gate is the
                # router (not a gate_proj), and _GATE_PROJ_ATTRS includes "gate" which
                # would incorrectly match the router module.
                if info.is_moe and hasattr(lc.mlp, "experts") and hasattr(lc.mlp, "gate"):
                    lc.moe_experts = lc.mlp.experts
                    lc.moe_router = lc.mlp.gate
                    lc.num_experts = info.num_experts
                    # Skip dense MLP projection lookup for MoE layers —
                    # experts hold gate_up_proj/down_proj as 3-D tensors, not nn.Linear
                else:
                    lc.gate_proj = _find_attr(lc.mlp, _GATE_PROJ_ATTRS)
                    lc.up_proj = _find_attr(lc.mlp, _UP_PROJ_ATTRS)
                    lc.down_proj = _find_attr(lc.mlp, _DOWN_PROJ_ATTRS)
                    lc.gate_up_proj = _find_attr(lc.mlp, _GATE_UP_FUSED_ATTRS)

            info.layer_components.append(lc)

    def _infer_family(self) -> str:
        """Infer the model family from config for informational purposes."""
        arch_list = getattr(self.config, "architectures", []) or []
        if arch_list:
            arch = arch_list[0].lower()
            for family in ["llama", "mistral", "qwen", "gemma", "phi", "deepseek", "olmoe"]:
                if family in arch:
                    return family
        model_type = getattr(self.config, "model_type", "")
        return model_type if model_type else "unknown"


def print_architecture_summary(info: ArchitectureInfo):
    """Print a readable summary of the analyzed architecture."""
    print(f"\n{'='*60}")
    print(f"  Architecture Analysis Results")
    print(f"{'='*60}")
    print(f"  Model Family     : {info.model_family}")
    print(f"  Hidden Size      : {info.hidden_size}")
    print(f"  Num Heads (Q)    : {info.num_heads}")
    print(f"  Num KV Heads     : {info.num_kv_heads}")
    print(f"  Head Dim         : {info.head_dim}")
    print(f"  GQA Groups       : {info.num_groups}")
    print(f"  Intermediate Size: {info.intermediate_size}")
    print(f"  Num Layers       : {info.num_layers}")
    print(f"  Vocab Size       : {info.vocab_size}")
    print(f"  ---")
    print(f"  Tied Embeddings  : {info.tie_word_embeddings}")
    print(f"  QK Norm          : {info.has_qk_norm}")
    print(f"  Post-FFW Norm    : {info.has_post_feedforward_layernorm}")
    print(f"  Fused QKV        : {info.has_fused_qkv}")
    print(f"  Fused Gate+Up    : {info.has_fused_gate_up}")
    print(f"  Norm Has Bias    : {info.norm_has_bias}")
    print(f"  Linear Has Bias  : {info.linear_has_bias}")
    print(f"  ---")
    print(f"  embed_tokens     : {'Found' if info.embed_tokens is not None else 'MISSING'}")
    print(f"  final_norm       : {'Found' if info.final_norm is not None else 'MISSING'}")
    print(f"  lm_head          : {'Found' if info.lm_head is not None else 'MISSING'}")
    print(f"  layers           : {len(info.layer_components)} layers parsed")

    if info.layer_components:
        lc = info.layer_components[0]
        found = []
        missing = []
        for name in ["input_layernorm", "post_attention_layernorm",
                      "q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]:
            if getattr(lc, name) is not None:
                found.append(name)
            else:
                missing.append(name)
        # Fused variants
        if lc.qkv_proj is not None:
            found.append("qkv_proj(fused)")
            # Remove q/k/v from missing since fused covers them
            for x in ["q_proj", "k_proj", "v_proj"]:
                if x in missing:
                    missing.remove(x)
        if lc.gate_up_proj is not None:
            found.append("gate_up_proj(fused)")
            for x in ["gate_proj", "up_proj"]:
                if x in missing:
                    missing.remove(x)
        # Optional extras
        for name, label in [("q_norm", "q_norm"), ("k_norm", "k_norm"),
                            ("pre_feedforward_layernorm", "pre_ffw_norm"),
                            ("post_feedforward_layernorm", "post_ffw_norm")]:
            if getattr(lc, name) is not None:
                found.append(label)

        print(f"  Layer[0] modules : {', '.join(found)}")
        if missing:
            print(f"  Layer[0] missing : {', '.join(missing)}")
    print(f"{'='*60}\n")
