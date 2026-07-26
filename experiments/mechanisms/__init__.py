"""Harness mechanisms.

Each module implements one harness mechanism from the research agenda (§2.2).
Mechanisms are switches: "off" is the bare ReAct baseline; "on" is the
scaffolding added here.

  info_seeking.py     – M1: Information-seeking (lead mechanism)
  memory.py           – M2: Memory / context
  state_ext.py        – M3: State externalization
  adaptive.py         – M4: Adaptive compute
  action_templating.py – M5: Action templating (secondary)
  planning.py         – M6: Planning / synthesis

Import from the specific module or from here:

    from mechanisms import InfoSeekingMechanism, MemoryMechanism
    from mechanisms.info_seeking import InfoSeekingMechanism
    from mechanisms.memory import MemoryMechanism
    from mechanisms.state_ext import StateExternalizer
    from mechanisms.adaptive import AdaptiveComputeController
    from mechanisms.action_templating import normalize_action
    from mechanisms.planning import PlanningMechanism
"""

from mechanisms.info_seeking import InfoSeekingMechanism
from mechanisms.memory import MemoryMechanism
from mechanisms.state_ext import StateExternalizer
from mechanisms.adaptive import AdaptiveComputeController
from mechanisms.action_templating import normalize_action
from mechanisms.planning import PlanningMechanism

__all__ = [
    'InfoSeekingMechanism',
    'MemoryMechanism',
    'StateExternalizer',
    'AdaptiveComputeController',
    'normalize_action',
    'PlanningMechanism',
]
