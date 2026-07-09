"""Failure-channel detectors.

Each module in this package implements a detector for one of the failure
channels defined in the research agenda (§2.1):

  ale.py    – Accepted-Local-Error  (ALE / M1 target)
  budget.py – Budget / Liveness     (repetition loops / M4 target)  [stub]
  drift.py  – Control-path drift    (goal loss / M2 target)          [stub]

Import from the specific module or from here:

    from channels import AcceptedLocalErrorDetector, make_env_checker
    from channels.ale import AcceptedLocalErrorDetector
    from channels.env_checker import make_env_checker
"""

from channels.ale import AcceptedLocalErrorDetector
from channels.env_checker import make_env_checker

__all__ = ['AcceptedLocalErrorDetector', 'make_env_checker']
