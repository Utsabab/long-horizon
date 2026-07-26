import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channels.ale import AcceptedLocalErrorDetector
from channels.budget import BudgetLivenessDetector
from channels.drift import DriftDetector, LLMCoherenceJudge
from channels.integration import IntegrationDetector, LLMIntegrationJudge
from mechanisms.action_templating import normalize_action
from mechanisms.adaptive import AdaptiveComputeController
from mechanisms.planning import PlanningMechanism


class FakeLLMClient:
    """Minimal test double for OpenRouterClient: generate() records the prompt
    and returns a canned response; parse_completion() unwraps it exactly like
    the real client's {content, tokens} shape."""

    def __init__(self, content, tokens=5):
        self.content = content
        self.tokens = tokens
        self.calls = []

    def generate(self, prompt, max_tokens=None, reasoning_enabled=False, **kwargs):
        self.calls.append(prompt)
        return {'content': self.content, 'tokens': self.tokens}

    def parse_completion(self, resp):
        return resp


class RaisingLLMClient:
    def generate(self, *args, **kwargs):
        raise RuntimeError('boom')

    def parse_completion(self, resp):
        return resp


# ---------------------------------------------------------------------------
# ale.py — repetition detector (unchanged pure logic)
# ---------------------------------------------------------------------------

def test_detector_triggers():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0)
    d.add('open door', 0)
    d.add('open door', 0)
    assert d.check_repetition()


def test_detector_no_trigger_on_progress():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0)
    d.add('open door', 0)
    d.add('open door', 1)
    assert not d.check_repetition()


# ---------------------------------------------------------------------------
# ale.py — check_ale() with a structured belief + mock env_checker
# ---------------------------------------------------------------------------

def test_check_ale_confirms_contradiction_from_structured_belief():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    belief = {'type': 'have_item', 'item': 'sword', 'location': None}
    d.add('attack troll with sword', 0, belief)
    d.add('attack troll with sword', 0, belief)
    d.add('attack troll with sword', 0, belief)

    def env_checker(belief_tuple):
        assert belief_tuple == ('have_item', 'sword')
        return True, {'reason': 'inventory_probe', 'probe_text': 'You are carrying nothing.'}

    is_ale, details = d.check_ale(env_checker)
    assert is_ale
    assert details['belief'] == ('have_item', 'sword')
    assert details['contradiction'] is True


def test_check_ale_no_repetition_short_circuits():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('go north', 0, {'type': 'have_item', 'item': 'sword'})

    def env_checker(belief_tuple):
        raise AssertionError('env_checker should not be called without repetition')

    is_ale, details = d.check_ale(env_checker)
    assert not is_ale
    assert details['reason'] == 'no_repetition'


def test_check_ale_no_belief_detected():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('wait', 0, None)
    d.add('wait', 0, None)
    d.add('wait', 0, None)

    def env_checker(belief_tuple):
        raise AssertionError('env_checker should not be called with no belief')

    is_ale, details = d.check_ale(env_checker)
    assert not is_ale
    assert details['reason'] == 'no_belief_detected'


# ---------------------------------------------------------------------------
# mechanisms/action_templating.py
# ---------------------------------------------------------------------------

def test_normalize_action_strips_label_prefixes():
    assert normalize_action('ACT: SEARCH ROAD') == 'SEARCH ROAD'
    assert normalize_action('ACTION: go north') == 'go north'
    assert normalize_action('COMMAND: open door') == 'open door'
    assert normalize_action('> go north') == 'go north'


def test_normalize_action_passthrough_when_no_prefix():
    assert normalize_action('open door') == 'open door'


def test_normalize_action_handles_empty():
    assert normalize_action('') == ''
    assert normalize_action(None) is None


# ---------------------------------------------------------------------------
# channels/budget.py — pure arithmetic, no LLM
# ---------------------------------------------------------------------------

def test_budget_liveness_below_threshold():
    b = BudgetLivenessDetector(max_steps=100, token_budget=None)
    b.add(0, 100, progress_increased=True)
    is_at_risk, details = b.check_budget()
    assert not is_at_risk
    assert details['reason'] == 'below_threshold'


def test_budget_liveness_flags_on_stall():
    b = BudgetLivenessDetector(max_steps=1000, token_budget=None,
                                stall_token_threshold=1000, threshold=0.3, w_stall=1.0,
                                w_step=0.0, w_token=0.0)
    for step in range(5):
        b.add(step, 300, progress_increased=False)
    is_at_risk, details = b.check_budget()
    assert is_at_risk
    assert details['tokens_since_progress'] == 1500
    assert details['stall_ratio'] == 1.0


def test_budget_liveness_resets_stall_on_progress():
    b = BudgetLivenessDetector(max_steps=1000, stall_token_threshold=1000)
    b.add(0, 900, progress_increased=False)
    b.add(1, 100, progress_increased=True)
    _, details = b.check_budget()
    assert details['tokens_since_progress'] == 0
    assert details['cumulative_tokens'] == 1000


# ---------------------------------------------------------------------------
# channels/integration.py — precheck (no LLM) + judge (fake client)
# ---------------------------------------------------------------------------

def test_integration_precheck_trips_on_unused_items():
    d = IntegrationDetector(min_unused=2, recent_window=10)
    tripped, unused = d.precheck(
        discoveries=['lamp', 'key'],
        inventory_items=['spell'],
        recent_actions=['go north', 'take lamp'],
    )
    assert tripped
    assert set(unused) == {'key', 'spell'}


def test_integration_precheck_does_not_trip_when_referenced():
    d = IntegrationDetector(min_unused=2, recent_window=10)
    tripped, unused = d.precheck(
        discoveries=['lamp', 'key'],
        inventory_items=[],
        recent_actions=['use lamp on key', 'go north'],
    )
    assert not tripped
    assert unused == []


def test_integration_judge_parses_gap_response():
    client = FakeLLMClient('{"gap": true, "suggested_combination": "unlock grate with key"}')
    judge = LLMIntegrationJudge()
    result, tokens = judge.check('escape the cellar', ['key', 'lamp'], ['go north'], client)
    assert result == {'gap': True, 'suggested_combination': 'unlock grate with key'}
    assert tokens == 5


def test_integration_judge_falls_back_on_llm_error():
    judge = LLMIntegrationJudge()
    result, tokens = judge.check('goal', ['key'], [], RaisingLLMClient())
    assert result == {'gap': False, 'suggested_combination': None}
    assert tokens == 0


# ---------------------------------------------------------------------------
# channels/drift.py — LLMCoherenceJudge + pure DriftDetector math
# ---------------------------------------------------------------------------

def test_coherence_judge_parses_score():
    client = FakeLLMClient('{"incoherence": 0.8}')
    judge = LLMCoherenceJudge()
    score, tokens = judge.score('escape', 'find key', ['go north'], 'I am looking around', client)
    assert score == 0.8
    assert tokens == 5


def test_coherence_judge_empty_reasoning_short_circuits():
    client = FakeLLMClient('{"incoherence": 0.8}')
    judge = LLMCoherenceJudge()
    score, tokens = judge.score('escape', 'find key', [], '', client)
    assert score == 0.5
    assert tokens == 0
    assert client.calls == []


def test_drift_detector_flags_high_composite_score():
    d = DriftDetector(window=3, threshold=0.5, w_stagnation=0.5, w_loop=0.5, w_coherence=0.0)
    d.add('look', 0, 0.0)
    d.add('look', 0, 0.0)
    d.add('look', 0, 0.0)
    is_drifting, details = d.check_drift()
    assert is_drifting
    # 3 identical actions -> 1 unique / 3 total -> loop_rate = 1 - 1/3
    assert details['loop_rate'] == round(1 - 1 / 3, 3)


def test_drift_detector_insufficient_history():
    d = DriftDetector(window=5)
    d.add('look', 0, 0.0)
    is_drifting, details = d.check_drift()
    assert not is_drifting
    assert details['reason'] == 'insufficient_history'


# ---------------------------------------------------------------------------
# mechanisms/adaptive.py — many-to-many risk-vector dispatch
# ---------------------------------------------------------------------------

def test_adaptive_ale_takes_priority():
    c = AdaptiveComputeController()
    bundle = c.decide({'drift_score': 0.9, 'ale_confirmed': True,
                        'budget_risk': 0.95, 'integration_precheck': True, 'confidence': 0.5})
    assert bundle['trigger_info_seek']
    assert not bundle['run_integration_judge']  # ALE dispatch takes priority this step


def test_adaptive_drift_alone_triggers_replan():
    c = AdaptiveComputeController(drift_replan_threshold=0.65)
    bundle = c.decide({'drift_score': 0.7, 'ale_confirmed': False,
                        'budget_risk': 0.0, 'integration_precheck': False, 'confidence': None})
    assert not bundle['trigger_info_seek']
    assert bundle['trigger_replan_from_drift']


def test_adaptive_skips_integration_judge_under_critical_budget():
    c = AdaptiveComputeController(skip_integration_judge_above_budget_risk=0.9)
    bundle = c.decide({'drift_score': 0.0, 'ale_confirmed': False,
                        'budget_risk': 0.95, 'integration_precheck': True, 'confidence': None})
    assert not bundle['run_integration_judge']
    assert bundle['budget_alert']


def test_adaptive_runs_integration_judge_when_precheck_trips_and_budget_ok():
    c = AdaptiveComputeController()
    bundle = c.decide({'drift_score': 0.0, 'ale_confirmed': False,
                        'budget_risk': 0.1, 'integration_precheck': True, 'confidence': None})
    assert bundle['run_integration_judge']
    assert not bundle['budget_alert']


# ---------------------------------------------------------------------------
# mechanisms/planning.py — replan (fake client)
# ---------------------------------------------------------------------------

class FakeMemory:
    def __init__(self):
        self.goal = 'escape the cellar'
        self.subgoal = 'find a light source'
        self.recent_actions = ['go north', 'take lamp']


def test_planning_replan_updates_memory_subgoal():
    client = FakeLLMClient('{"subgoal": "unlock the grate with the key then go down"}')
    memory = FakeMemory()
    planner = PlanningMechanism()
    new_subgoal, tokens = planner.replan(
        memory.goal, memory, {'location': 'cellar', 'inventory_items': ['key']}, client,
        suggested_combination='use key on grate',
    )
    assert new_subgoal == 'unlock the grate with the key then go down'
    assert memory.subgoal == new_subgoal
    assert tokens == 5


def test_planning_replan_returns_none_on_unusable_response():
    client = FakeLLMClient('not json at all')
    memory = FakeMemory()
    planner = PlanningMechanism()
    new_subgoal, tokens = planner.replan(memory.goal, memory, None, client)
    assert new_subgoal is None
    assert memory.subgoal == 'find a light source'  # left unchanged


def test_planning_replan_handles_llm_failure():
    memory = FakeMemory()
    planner = PlanningMechanism()
    new_subgoal, tokens = planner.replan(memory.goal, memory, None, RaisingLLMClient())
    assert new_subgoal is None
    assert tokens == 0
