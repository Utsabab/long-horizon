"""Failure-channel detectors.

Each module in this package implements a detector for one of the failure
channels defined in the research agenda (§2.1):

  ale.py         – Accepted-Local-Error  (ALE / M1 primary target)
  drift.py       – Control-path drift    (goal loss / M2 primary target)
  budget.py      – Budget / Liveness     (loops/stalls / M4 primary target)
  integration.py – Integration / synthesis (can't combine known pieces / M6 primary target)

Import from the specific module or from here:

    from channels import AcceptedLocalErrorDetector, DriftDetector, make_env_checker
    from channels.ale import AcceptedLocalErrorDetector
    from channels.drift import DriftDetector, LLMCoherenceJudge
    from channels.budget import BudgetLivenessDetector
    from channels.integration import IntegrationDetector, LLMIntegrationJudge
    from channels.env_checker import make_env_checker
"""

from channels.ale import AcceptedLocalErrorDetector
from channels.drift import DriftDetector, LLMCoherenceJudge
from channels.budget import BudgetLivenessDetector
from channels.integration import IntegrationDetector, LLMIntegrationJudge
from channels.env_checker import make_env_checker

__all__ = [
    'AcceptedLocalErrorDetector',
    'DriftDetector',
    'LLMCoherenceJudge',
    'BudgetLivenessDetector',
    'IntegrationDetector',
    'LLMIntegrationJudge',
    'make_env_checker',
]
