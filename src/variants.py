"""Model selection shared by source runs, the exporter, and the botpack executable."""
from dataclasses import dataclass
from pathlib import Path
import os
import sys

ASSETS = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1])) / 'bots'


@dataclass(frozen=True)
class Variant:
    name: str
    agent_id: str
    policy: str
    base: str | None


VARIANTS = {
    'v1.1': Variant('Ripple v1.1', 'rolv/ripple/1.1', 'v1.1/ripple-v1.1', 'v1.1/ripple-v1.1-bcm'),
    'v1.0': Variant('Ripple v1.0', 'rolv/ripple/1.0', 'v1.0/ripple-v1.0', 'v1.0/ripple-v1.0-bcm'),
}


def select_variant(name=None, agent_id=None):
    if name is not None:
        return VARIANTS[name]
    agent_id = os.environ.get('RLBOT_AGENT_ID') if agent_id is None else agent_id
    if not agent_id:
        return VARIANTS['v1.1']
    for variant in VARIANTS.values():
        if variant.agent_id == agent_id:
            return variant
    raise ValueError(f'Unknown Ripple agent id: {agent_id!r}')


def model_paths(variant, backend, bcm_only=False):
    suffix = '.onnx' if backend == 'onnx' else '.pt'
    policy = ASSETS / (variant.policy + suffix)
    base = ASSETS / (variant.base + suffix) if variant.base else None
    if bcm_only:
        if base is None:
            raise ValueError(f'{variant.name} has no imitation model')
        return base, None
    return policy, base


def source_models():
    """Each original model once, shared by conversion and validation tools."""
    return tuple(ASSETS / (stem + '.pt') for variant in VARIANTS.values()
                 for stem in (variant.policy, variant.base) if stem is not None)
