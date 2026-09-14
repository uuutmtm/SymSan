"""Verifies logit equivalence after the defense."""

import torch
import torch.nn as nn
from typing import Optional
from .architecture import ArchitectureInfo


class EquivalenceVerifier:
    """Verifies that symmetry transforms preserve the model output."""

    def __init__(self, model: nn.Module, info: ArchitectureInfo):
        self.model = model
        self.info = info

    # ========================================================================
    # Pre-transform: Snapshot
    # ========================================================================

    def snapshot_output(self, tokenizer, test_text: str = "The capital of France is",
                        max_length: int = 20) -> Optional[torch.Tensor]:
        """Run a forward pass and save logits for comparison."""
        if tokenizer is None:
            return None

        self.model.eval()
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Gemma3ForCausalLM requires token_type_ids during forward; retry with zeros if no logits
            if not hasattr(outputs, 'logits') or outputs.logits is None:
                token_type_ids = torch.zeros_like(input_ids)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        if not hasattr(outputs, 'logits') or outputs.logits is None:
            return None
        return outputs.logits.cpu()

    # ========================================================================
    # Post-transform: Verify
    # ========================================================================

    def verify_equivalence(self, ref_logits: Optional[torch.Tensor], tokenizer,
                           test_text: str = "The capital of France is") -> dict:
        """
        Compare post-transform logits against reference.
        Returns a dict with max_diff, mean_diff, and pass/fail status.
        """
        if ref_logits is None or tokenizer is None:
            return {"status": "SKIPPED", "reason": "no reference logits or tokenizer"}

        new_logits = self.snapshot_output(tokenizer, test_text)
        if new_logits is None:
            return {"status": "ERROR", "reason": "failed to compute new logits"}

        diff = (ref_logits.float() - new_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Primary check: top-1 token agreement (strongest equivalence test)
        ref_tokens = ref_logits.argmax(dim=-1)
        new_tokens = new_logits.argmax(dim=-1)
        tokens_match = torch.equal(ref_tokens, new_tokens)

        # Secondary check: logit tolerance (for information)
        # Float16 accumulates ~O(L * eps) error over L layers.
        # bfloat16 has much less mantissa precision (7 bits vs 10), so error is larger.
        dtype = next(self.model.parameters()).dtype
        if dtype == torch.bfloat16:
            tol = 1.0  # bfloat16: ~7.8e-3 eps, 30+ layers → expect ~0.5-1.0
        elif dtype == torch.float16:
            tol = 2e-1  # float16: ~4.9e-4 eps, 30+ layers → expect ~0.05-0.15
        else:
            tol = 1e-4

        # Pass if tokens match AND logit diff is within tolerance
        passed = tokens_match and max_diff < tol

        return {
            "status": "PASS" if passed else "FAIL",
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "tolerance": tol,
            "dtype": str(dtype),
            "tokens_match": tokens_match,
        }


def print_verification_results(equiv_result: dict):
    """Print formatted equivalence results to terminal."""
    print(f"\n{'='*60}")
    print(f"  Verification Results")
    print(f"{'='*60}")

    print(f"\n  [Functional Equivalence]")
    if equiv_result.get("status") == "PASS":
        print(f"    Status       : PASS")
        print(f"    Tokens Match : {equiv_result.get('tokens_match', 'N/A')}")
        print(f"    Max Logit Diff: {equiv_result['max_diff']:.2e}")
        print(f"    Mean Logit Diff: {equiv_result['mean_diff']:.2e}")
        print(f"    Tolerance    : {equiv_result['tolerance']:.2e} ({equiv_result['dtype']})")
    elif equiv_result.get("status") == "FAIL":
        print(f"    Status       : FAIL")
        print(f"    Tokens Match : {equiv_result.get('tokens_match', 'N/A')}")
        print(f"    Max Logit Diff: {equiv_result['max_diff']:.2e}")
        print(f"    Tolerance    : {equiv_result['tolerance']:.2e}")
    else:
        print(f"    Status       : {equiv_result.get('status', 'UNKNOWN')}")
        if "reason" in equiv_result:
            print(f"    Reason       : {equiv_result['reason']}")

    print(f"{'='*60}\n")
