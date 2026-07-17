"""Harness mechanisms.

Each module implements one harness mechanism from the research agenda (§2.2).
Mechanisms are switches: "off" is the bare ReAct baseline; "on" is the
scaffolding added here.

  info_seeking.py  – M1: Information-seeking (lead mechanism)
  memory.py        – M2: Memory / context
  state_ext.py     – M3: State externalization                 [stub]
  adaptive.py      – M4: Adaptive compute                      [stub]
  planning.py      – M6: Planning / synthesis                  [stub]

Import from the specific module or from here:

    from mechanisms import InfoSeekingMechanism, MemoryMechanism
    from mechanisms.info_seeking import InfoSeekingMechanism
    from mechanisms.memory import MemoryMechanism
"""

from mechanisms.info_seeking import InfoSeekingMechanism
from mechanisms.memory import MemoryMechanism

__all__ = ['InfoSeekingMechanism', 'MemoryMechanism']
