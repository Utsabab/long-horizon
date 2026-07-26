"""M4: Adaptive compute mechanism (research agenda §2.2).

Budget/liveness channel: the agent spends compute badly. M4's primary
detector is channels.budget.BudgetLivenessDetector (pure arithmetic on
steps/tokens/stall).

M4 also serves a second role: it is the shared, composable policy layer that
turns per-channel risk signals into a mechanism-bundle decision each step —
replacing the old hardcoded "ALE confirmed -> M1, nothing else" branch in
runner.py with a many-to-many dispatch. A single high-drift-risk signal can
trigger a replan (M6) with no ALE and no confirmed integration gap; an
integration pre-filter hit only spends its (costly) LLM judge call when the
remaining budget allows it.

Nothing here calls an LLM or parses text — it is a deterministic function of
already-computed numeric/boolean signals from the other channels.
"""


class AdaptiveComputeController:
    """Turn a per-step risk vector into a bundle of mechanism triggers."""

    def __init__(self, drift_replan_threshold=0.65, budget_alert_threshold=0.7,
                 skip_integration_judge_above_budget_risk=0.9):
        """
        Args:
            drift_replan_threshold: drift_score at or above which sustained
                drift alone (no ALE) triggers an M6 replan.
            budget_alert_threshold: budget_risk at or above which
                budget_alert is raised (logged, not itself an intervention).
            skip_integration_judge_above_budget_risk: budget_risk at or above
                which the (costly) integration LLM judge is skipped even if
                its cheap pre-filter tripped, to conserve remaining budget.
        """
        self.drift_replan_threshold = drift_replan_threshold
        self.budget_alert_threshold = budget_alert_threshold
        self.skip_integration_judge_above_budget_risk = skip_integration_judge_above_budget_risk

    def decide(self, risk_vector):
        """Return a mechanism bundle for this step.

        Args:
            risk_vector: dict with keys
                drift_score: float in [0, 1] (DriftDetector.drift_score(), or 0.0)
                ale_confirmed: bool (AcceptedLocalErrorDetector.check_ale() result)
                budget_risk: float in [0, 1] (BudgetLivenessDetector.check_budget())
                integration_precheck: bool (IntegrationDetector.precheck() tripped)
                confidence: float or None (model's self-reported confidence;
                    currently informational/loggable, not yet gating a
                    mechanism on its own)

        Returns:
            dict: {
              'trigger_info_seek': bool,          -- M1 should intervene this step
              'run_integration_judge': bool,      -- worth spending the M6-precursor LLM call
              'trigger_replan_from_drift': bool,  -- secondary M6 trigger, independent of integration
              'budget_alert': bool,               -- logged budget/liveness risk flag
            }
        """
        drift_score = risk_vector.get('drift_score') or 0.0
        ale_confirmed = bool(risk_vector.get('ale_confirmed'))
        budget_risk = risk_vector.get('budget_risk') or 0.0
        integration_precheck = bool(risk_vector.get('integration_precheck'))

        budget_alert = budget_risk >= self.budget_alert_threshold

        # M1 is the lead mechanism: a confirmed ALE always takes priority
        # over everything else this step.
        trigger_info_seek = ale_confirmed

        # Skip the integration judge's LLM call once budget risk is critical,
        # even if the cheap pre-filter tripped — M4 gating the cost of M6.
        run_integration_judge = (
            integration_precheck
            and not trigger_info_seek
            and budget_risk < self.skip_integration_judge_above_budget_risk
        )

        # Secondary trigger: sustained high drift alone (no ALE this step,
        # regardless of whether the integration pre-filter tripped) still
        # warrants a replan — the many-to-many path where drift also feeds M6.
        trigger_replan_from_drift = (
            not trigger_info_seek and drift_score >= self.drift_replan_threshold
        )

        return {
            'trigger_info_seek': trigger_info_seek,
            'run_integration_judge': run_integration_judge,
            'trigger_replan_from_drift': trigger_replan_from_drift,
            'budget_alert': budget_alert,
        }
