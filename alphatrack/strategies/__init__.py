"""Strategy registry. Register plugins implementing prepare(frame) and decide(row)."""
from .fundd import FunddStrategy

STRATEGIES = {"fundd": FunddStrategy}


def register_strategy(name, factory):
    if name in STRATEGIES:
        raise ValueError(f"Strategy already registered: {name}")
    STRATEGIES[name] = factory


def create_strategy(name, params=None):
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; available: {list(STRATEGIES)}")
    return STRATEGIES[name](**(params or {}))
