def is_goal(obs, threshold: float = 0.98) -> bool:
    """Returns True when every line loading is below the safety threshold."""
    return float(obs.rho.max()) < threshold
