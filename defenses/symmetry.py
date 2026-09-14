"""SymSan defense entry point: one-call API around SymmetryDefenseEngine."""

from .symmetry_core.engine import SymmetryDefenseEngine


def _coerce_bool(v, default=True):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v) if v is not None else default


def run_symmetry_defense(model, tokenizer, **kwargs):
    """Sanitize `model` in place with SymSan and verify functional equivalence.

    kwargs: seed, scale_range, verify, verbose.

    Returns (model, report), where `report` is a dict with timing, the list of
    applied transforms, and (if verify=True) the equivalence result — see
    SymmetryDefenseEngine.last_report.
    """
    seed = int(kwargs.get("seed", 42))
    scale_range = float(kwargs.get("scale_range", 0.05))
    verify = _coerce_bool(kwargs.get("verify", True), True)
    verbose = _coerce_bool(kwargs.get("verbose", True), True)

    engine = SymmetryDefenseEngine(
        model=model,
        tokenizer=tokenizer,
        seed=seed,
        scale_range=scale_range,
        verify=verify,
        verbose=verbose,
    )

    model = engine.run()
    return model, engine.last_report
