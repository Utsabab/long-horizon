"""Experiment runner.

A Runner manages one game + one LLM client + one harness configuration.
Call run_once() to execute a single episode and get back a run_id that can
be used to look up the transcript written to out_dir.

Run directly to execute one episode with the settings in config.yaml:

    cd experiments
    python runner.py

Or specify a game name as an argument:

    python runner.py wishbringer
"""

import os
import sys
import uuid
from pathlib import Path

import yaml

# Ensure the experiments/ directory is in sys.path so sub-packages resolve
# whether the script is run as 'python runner.py' from inside experiments/
# or as 'python experiments/runner.py' from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agent import build_step_messages, format_history
from channels import (
    AcceptedLocalErrorDetector,
    BudgetLivenessDetector,
    DriftDetector,
    IntegrationDetector,
    LLMCoherenceJudge,
    LLMIntegrationJudge,
    make_env_checker,
)
from env import find_game_path, load_env_class, resolve_root
from llm import OpenRouterClient, extract_step_fields
from logger import RunLogger
from mechanisms import (
    AdaptiveComputeController,
    InfoSeekingMechanism,
    MemoryMechanism,
    PlanningMechanism,
    StateExternalizer,
    normalize_action,
)


class Runner:
    """Run one game episode under a given harness configuration.

    Parameters
    ----------
    game_name : str
        Name of the TextQuests game (e.g. 'zork1', 'wishbringer').
    out_dir : Path-like
        Directory where per-run JSONL logs are written.
    llm_client : OpenRouterClient
        Pre-configured LLM client.
    config : dict
        Experiment config dict (from the 'experiments' key in config.yaml).
    textquests_root : Path-like, optional
        Override for the textquests checkout path. Defaults to the submodule.
    """

    def __init__(self, game_name, out_dir, llm_client, config, textquests_root=None):
        self.game_name = game_name
        self.out_dir = Path(out_dir)
        self.max_steps = config.get('max_steps', 500)
        self.ale_k = config.get('accepted_local_error_k', 3)
        self.ale_window = config.get('info_seek_window', 10)
        self.textquests_root = (
            Path(textquests_root).expanduser().resolve() if textquests_root else None
        )

        drift_cfg = config.get('drift', {}) or {}
        self.drift_enabled      = drift_cfg.get('enabled', True)
        self.drift_window       = drift_cfg.get('window', 20)
        self.drift_threshold    = drift_cfg.get('threshold', 0.65)
        self.drift_w_stagnation = drift_cfg.get('w_stagnation', 0.5)
        self.drift_w_loop       = drift_cfg.get('w_loop', 0.3)
        self.drift_w_coherence  = drift_cfg.get('w_coherence', 0.2)

        mem_cfg = config.get('memory', {}) or {}
        self.memory_max_visited     = mem_cfg.get('max_visited', 25)
        self.memory_max_discoveries = mem_cfg.get('max_discoveries', 10)

        budget_cfg = config.get('budget', {}) or {}
        self.budget_enabled        = budget_cfg.get('enabled', True)
        self.budget_token_budget   = budget_cfg.get('token_budget')
        self.budget_stall_tokens   = budget_cfg.get('stall_token_threshold', 4000)
        self.budget_w_step         = budget_cfg.get('w_step', 0.3)
        self.budget_w_token        = budget_cfg.get('w_token', 0.3)
        self.budget_w_stall        = budget_cfg.get('w_stall', 0.4)
        self.budget_threshold      = budget_cfg.get('threshold', 0.7)

        integration_cfg = config.get('integration', {}) or {}
        self.integration_enabled       = integration_cfg.get('enabled', True)
        self.integration_min_unused    = integration_cfg.get('min_unused', 2)
        self.integration_recent_window = integration_cfg.get('recent_window', 10)

        mech_cfg = config.get('mechanisms', {}) or {}
        self.m1_enabled = mech_cfg.get('m1_info_seeking', True)
        self.m2_enabled = mech_cfg.get('m2_memory', True)
        self.m3_enabled = mech_cfg.get('m3_state_ext', True)
        self.m4_enabled = mech_cfg.get('m4_adaptive', True)
        self.m5_enabled = mech_cfg.get('m5_action_templating', True)
        self.m6_enabled = mech_cfg.get('m6_planning', True)

        self.logger = RunLogger(self.out_dir)
        self.llm = llm_client
        self._env_class = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_env_class(self):
        if self._env_class is None:
            self._env_class = load_env_class(self.textquests_root)
        return self._env_class

    def _new_env(self, seed):
        game_path = find_game_path(self.game_name, self.textquests_root)
        return self._get_env_class()(str(game_path), seed=seed)

    def _build_mechanisms(self):
        """Instantiate the per-episode detector/mechanism set from config flags.

        Returns a dict of optional components; any entry is None when its
        governing mechanism/channel is disabled, so _run_loop only has to
        check truthiness rather than re-reading config.
        """
        mech = {}

        mech['drift_detector'] = DriftDetector(
            window=self.drift_window,
            threshold=self.drift_threshold,
            w_stagnation=self.drift_w_stagnation,
            w_loop=self.drift_w_loop,
            w_coherence=self.drift_w_coherence,
        ) if self.drift_enabled else None
        mech['coherence_judge'] = LLMCoherenceJudge() if self.drift_enabled else None

        mech['memory'] = MemoryMechanism(
            max_visited=self.memory_max_visited,
            max_discoveries=self.memory_max_discoveries,
        ) if self.m2_enabled else None

        mech['state_ext'] = StateExternalizer() if self.m3_enabled else None

        mech['budget_detector'] = BudgetLivenessDetector(
            max_steps=self.max_steps,
            token_budget=self.budget_token_budget,
            stall_token_threshold=self.budget_stall_tokens,
            w_step=self.budget_w_step,
            w_token=self.budget_w_token,
            w_stall=self.budget_w_stall,
            threshold=self.budget_threshold,
        ) if self.budget_enabled else None

        mech['integration_detector'] = IntegrationDetector(
            min_unused=self.integration_min_unused,
            recent_window=self.integration_recent_window,
        ) if self.integration_enabled else None
        mech['integration_judge'] = LLMIntegrationJudge() if self.integration_enabled else None

        mech['adaptive'] = AdaptiveComputeController() if self.m4_enabled else None
        mech['planning'] = PlanningMechanism(window=self.ale_window) if self.m6_enabled else None

        return mech

    def _run_loop(self, env, logger, run_id, history, observation, info, progress, score,
                  start_step, detector, env_checker, mech, info_seeking=None, capture_fork=False):
        """Run steps [start_step, self.max_steps) on a live env.

        Per-step pipeline:
          1. Check for a confirmed Accepted-Local-Error (channel, always
             evaluated) and the cheap integration/budget signals (channels).
          2. Ask mech['adaptive'] (M4) to turn those signals into a mechanism
             bundle for this step (many-to-many: e.g. sustained drift alone
             can trigger a replan with no ALE and no integration gap).
          3. Dispatch: M1 intervention (lead, takes priority) > M6 replan
             (integration-gap primary, high-drift secondary) > normal
             generation. M5 normalizes whatever action results.
          4. Update M2/M3/drift/budget bookkeeping and log the step.

        Returns a dict: observation, progress, score, game_over, last_step,
        history, fork (None unless capture_fork captured one).
        """
        memory              = mech.get('memory')
        state_ext           = mech.get('state_ext')
        drift_detector       = mech.get('drift_detector')
        coherence_judge      = mech.get('coherence_judge')
        budget_detector      = mech.get('budget_detector')
        integration_detector = mech.get('integration_detector')
        integration_judge    = mech.get('integration_judge')
        adaptive             = mech.get('adaptive')
        planning             = mech.get('planning')

        game_over = False
        last_checkpoint_id = None
        fork = None
        step = start_step - 1

        current_goal = memory.goal if memory else None
        current_subgoal = memory.subgoal if memory else None
        last_confidence = None
        state = state_ext.probe(env) if state_ext else None

        for step in range(start_step, self.max_steps):
            is_ale, ale_details = detector.check_ale(env_checker)
            used_intervention = False
            mechanisms_used = []
            judge_tokens = 0
            action = ''
            reasoning = ''
            tokens = 0
            belief = None
            confidence = last_confidence

            if is_ale:
                logger.add_ale({'step': step, **ale_details})
                if capture_fork and fork is None:
                    fork = {
                        'step': step,
                        'checkpoint_id': last_checkpoint_id,
                        'progress': progress,
                        'score': score,
                        'steps_snapshot': list(logger._steps),
                        'milestones_snapshot': list(logger._milestones),
                    }

            # --- Cheap channel signals used by M4's dispatch (§integration/budget) ---
            integration_tripped, unused_items = (False, [])
            if integration_detector is not None:
                integration_tripped, unused_items = integration_detector.precheck(
                    [],
                    memory.inventory_items if memory else [],
                    memory.recent_actions if memory else [],
                )

            budget_risk = 0.0
            budget_preview = None
            if budget_detector is not None:
                _, budget_preview = budget_detector.check_budget()
                budget_risk = budget_preview.get('budget_risk', 0.0)

            drift_score = drift_detector.drift_score() if drift_detector else 0.0

            risk_vector = {
                'drift_score': drift_score,
                'ale_confirmed': is_ale,
                'budget_risk': budget_risk,
                'integration_precheck': integration_tripped,
                'confidence': confidence,
            }

            if adaptive is not None:
                bundle = adaptive.decide(risk_vector)
            else:
                bundle = {
                    'trigger_info_seek': is_ale,
                    'run_integration_judge': False,
                    'trigger_replan_from_drift': False,
                    'budget_alert': False,
                }

            if bundle['budget_alert'] and budget_preview is not None:
                logger.add_budget_event({'step': step, **budget_preview})

            # --- Dispatch: M1 (lead) ---
            if bundle['trigger_info_seek'] and info_seeking is not None:
                try:
                    proposed_action, parsed = info_seeking.intervene(
                        format_history(history), ale_details['belief'], ale_details['evidence'],
                    )
                except Exception as exc:
                    logger.add_error({'step': step, 'error': f'info_seeking.intervene failed: {exc}'})
                    proposed_action = ''
                if proposed_action:
                    action = proposed_action
                    reasoning = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
                    tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0
                    used_intervention = True
                    mechanisms_used.append('m1_info_seeking')

            # --- Dispatch: M6 (integration-gap primary, drift secondary) ---
            replanned = False
            if not used_intervention and bundle['run_integration_judge'] and integration_judge is not None:
                result, itoks = integration_judge.check(
                    current_subgoal, unused_items,
                    memory.recent_actions if memory else [], self.llm,
                )
                judge_tokens += itoks
                if result['gap']:
                    logger.add_integration_event({'step': step, 'unused': unused_items, **result})
                    mechanisms_used.append('m6_integration_gap')
                    if planning is not None:
                        new_subgoal, rtoks = planning.replan(
                            current_goal, memory, state, self.llm,
                            suggested_combination=result['suggested_combination'],
                        )
                        judge_tokens += rtoks
                        if new_subgoal:
                            current_subgoal = new_subgoal
                            replanned = True
                            mechanisms_used.append('m6_planning')

            if not used_intervention and not replanned and bundle['trigger_replan_from_drift'] and planning is not None:
                new_subgoal, rtoks = planning.replan(current_goal, memory, state, self.llm)
                judge_tokens += rtoks
                if new_subgoal:
                    current_subgoal = new_subgoal
                    replanned = True
                    mechanisms_used.append('m6_planning_drift')

            # --- Dispatch: normal generation ---
            if not used_intervention:
                scratchpad = memory.render() if memory else None
                messages = build_step_messages(self.game_name, history, observation,
                                               scratchpad=scratchpad)
                try:
                    resp = self.llm.generate(messages, max_tokens=192, reasoning_enabled=False)
                    parsed = self.llm.parse_completion(resp)
                except Exception as exc:
                    logger.add_error({'step': step, 'error': f'llm.generate failed: {exc}'})
                    break
                content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
                tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0
                fields = extract_step_fields(content)
                action = fields['action']
                reasoning = fields['reasoning']
                if fields['goal']:
                    current_goal = fields['goal']
                if fields['subgoal']:
                    current_subgoal = fields['subgoal']
                belief = fields['belief']
                confidence = fields['confidence']
                if memory:
                    memory.set_goal(current_goal or memory.goal)

            # --- M5: action templating guardrail ---
            if self.m5_enabled:
                action = normalize_action(action)

            try:
                new_observation, reward, game_over, info = env.step(action)
            except Exception as exc:
                logger.add_error({'step': step, 'error': str(exc)})
                break

            new_progress = info.get('game_progress', progress) if isinstance(info, dict) else progress
            score = info.get('score', 0) if isinstance(info, dict) else 0
            progress_increased = new_progress > progress

            # --- M3: ground-truth post-step probe ---
            new_state = state_ext.probe(env) if state_ext else None
            action_failed = bool(
                state_ext and state and new_state
                and state.get('location') == new_state.get('location')
                and state.get('inventory_items') == new_state.get('inventory_items')
            )

            # --- Drift score (channel; incoherence scored by LLMCoherenceJudge) ---
            # Computed before logging so the step's 'tokens' total honestly
            # includes this judge call's cost, not just next step's.
            incoherence_score = 0.0
            if drift_detector:
                recent_actions = [h['action'] for h in history[-5:]]
                if coherence_judge is not None:
                    incoherence_score, ctoks = coherence_judge.score(
                        current_goal, current_subgoal, recent_actions, reasoning, self.llm,
                    )
                    judge_tokens += ctoks
                else:
                    incoherence_score = 0.0

            total_tokens = tokens + judge_tokens

            logger.add_step({
                'step': step,
                'obs': observation,
                'action': action,
                'reasoning': reasoning,
                'progress': new_progress,
                'score': score,
                'reward': reward,
                'tokens': total_tokens,
                'judge_tokens': judge_tokens,
                'goal': current_goal,
                'subgoal': current_subgoal,
                'belief': belief,
                'confidence': confidence,
                'mechanisms_used': mechanisms_used,
            })
            history.append({'obs': observation, 'action': action, 'reasoning': reasoning, 'progress': new_progress})

            if new_progress > progress:
                logger.add_milestone({
                    'step': step,
                    'progress_before': progress,
                    'progress_after': new_progress,
                    'score': score,
                    'action': action,
                })
                print(f'[{run_id[:8]}] step {step:>4}  {progress:.1f}% → {new_progress:.1f}%  score {score}  action: {action}')

            # --- M2: update memory scratchpad from M3 ground truth ---
            if memory:
                memory.update(
                    new_state.get('location') if new_state else None,
                    new_state.get('inventory_items') if new_state else None,
                    action, action_failed,
                    goal=current_goal, subgoal=current_subgoal,
                )

            # --- Drift detection (channel) ---
            if drift_detector:
                drift_detector.add(action, new_progress, incoherence_score)
                is_drifting, drift_details = drift_detector.check_drift()
                if is_drifting:
                    logger.add_drift_event({'step': step, **drift_details})
                    print(f'[{run_id[:8]}] step {step:>4}  DRIFT  score={drift_details["drift_score"]:.2f}'
                          f'  (stagnation={drift_details["stagnation"]:.2f}'
                          f'  loop={drift_details["loop_rate"]:.2f}'
                          f'  incoherence={drift_details["incoherence"]:.2f})')

            # --- Budget/liveness bookkeeping (channel) ---
            if budget_detector:
                budget_detector.add(step, total_tokens, progress_increased)

            try:
                last_checkpoint_id = env.save_checkpoint(new_observation, info)
            except Exception:
                pass

            detector.add(action, new_progress, belief)

            if new_state is not None:
                state = new_state
            last_confidence = confidence

            if game_over:
                break

            observation = new_observation
            progress = new_progress

        return {
            'observation': observation,
            'progress': progress,
            'score': score,
            'game_over': game_over,
            'last_step': step,
            'history': history,
            'fork': fork,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self, seed=0, label='baseline'):
        """Run one episode with the harness configured in config.yaml.

        Args:
            label: recorded in the transcript summary and used by plot.py to
                distinguish lines when comparing multiple runs.

        Returns
        -------
        str
            The run_id (hex string). Transcript written to out_dir/<run_id>.json.
        """
        run_id = uuid.uuid4().hex
        env = self._new_env(seed)
        observation, info = env.reset()
        detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        env_checker = make_env_checker(env)

        mech = self._build_mechanisms()
        if mech['memory']:
            mech['memory'].update('', [], '', False)

        info_seeking = InfoSeekingMechanism(self.llm, window=self.ale_window) if self.m1_enabled else None

        result = self._run_loop(
            env, self.logger, run_id, [], observation, info, 0, 0, 0,
            detector, env_checker, mech, info_seeking=info_seeking, capture_fork=False,
        )

        path = self.logger.save(run_id, self.game_name, {
            'total_steps': result['last_step'] + 1,
            'final_progress': result['progress'],
            'final_score': result['score'],
            'game_over': result['game_over'],
            'label': label,
            'fork_step': None,
        })
        print(f'[{run_id[:8]}] done — {result["last_step"] + 1} steps, progress {result["progress"]:.1f}%  →  {path}')

        return run_id

    def run_comparison(self, seed=0):
        """Run a baseline episode, and if an ALE is detected, fork a harness episode.

        The baseline (control) plays the whole episode with M1 (info-seeking)
        disabled. If an Accepted-Local-Error is confirmed at some step, the env
        is rewound (via restore_checkpoint) to the decision point right before
        that step, and a second episode (harness/treatment) is played from
        there with InfoSeekingMechanism intervening whenever an ALE is
        confirmed, instead of letting the model act on the flagged belief
        unchecked. All other configured mechanisms (M2–M6) run identically on
        both branches, so the only variable being compared is M1 itself.

        Returns
        -------
        (baseline_run_id, harness_run_id_or_None, fork_step_or_None)
        """
        baseline_id = uuid.uuid4().hex
        env = self._new_env(seed)
        observation, info = env.reset()
        detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        env_checker = make_env_checker(env)

        baseline_mech = self._build_mechanisms()
        if baseline_mech['memory']:
            baseline_mech['memory'].update('', [], '', False)

        result = self._run_loop(
            env, self.logger, baseline_id, [], observation, info, 0, 0, 0,
            detector, env_checker, baseline_mech, info_seeking=None, capture_fork=True,
        )

        baseline_path = self.logger.save(baseline_id, self.game_name, {
            'total_steps': result['last_step'] + 1,
            'final_progress': result['progress'],
            'final_score': result['score'],
            'game_over': result['game_over'],
            'label': 'baseline',
            'fork_step': None,
        })
        print(f'[{baseline_id[:8]}] baseline done — {result["last_step"] + 1} steps, '
              f'progress {result["progress"]:.1f}%  →  {baseline_path}')

        fork = result['fork']
        if fork is None:
            print(f'No ALE detected during baseline run of {self.game_name}; nothing to fork.')
            return baseline_id, None, None

        obs, ckpt_info, ok = env.restore_checkpoint(fork['checkpoint_id'])
        if not ok:
            print('Checkpoint restore failed; cannot run harness branch.')
            return baseline_id, None, fork['step']

        harness_detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        for a, p, b in zip(detector.actions, detector.progress, detector.beliefs):
            harness_detector.add(a, p, b)

        info_seeking = InfoSeekingMechanism(self.llm, window=self.ale_window)
        harness_mech = self._build_mechanisms()

        for step_record in fork['steps_snapshot']:
            self.logger.add_step(step_record)
        for milestone_record in fork['milestones_snapshot']:
            self.logger.add_milestone(milestone_record)

        harness_id = uuid.uuid4().hex
        harness_history = result['history'][:fork['step']]

        harness_result = self._run_loop(
            env, self.logger, harness_id, harness_history, obs, ckpt_info,
            fork['progress'], fork['score'], fork['step'],
            harness_detector, env_checker, harness_mech, info_seeking=info_seeking, capture_fork=False,
        )

        harness_path = self.logger.save(harness_id, self.game_name, {
            'total_steps': harness_result['last_step'] + 1,
            'final_progress': harness_result['progress'],
            'final_score': harness_result['score'],
            'game_over': harness_result['game_over'],
            'label': 'harness',
            'fork_step': fork['step'],
        })
        print(f'[{harness_id[:8]}] harness done — {harness_result["last_step"] + 1} steps, '
              f'progress {harness_result["progress"]:.1f}%  →  {harness_path}')

        return baseline_id, harness_id, fork['step']


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _load_config():
    cfg_path = _HERE / 'config.yaml'
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _build_runner(cfg, game_name=None, mechanism_overrides=None):
    """Construct a Runner from a config dict.

    Args:
        mechanism_overrides: optional dict merged into cfg['experiments']['mechanisms'],
            letting callers (e.g. run_mechanism_comparison.py) toggle individual
            mechanisms without editing config.yaml.
    """
    textquests_root = resolve_root(cfg)
    or_cfg = cfg.get('openrouter', {}) or {}
    api_key = (
        or_cfg.get('api_key')
        or or_cfg.get('OPENROUTER_API_KEY')
        or os.getenv('OPENROUTER_API_KEY')
    )
    llm = OpenRouterClient(
        api_key=api_key,
        model=or_cfg.get('model'),
        base_url=or_cfg.get('base_url'),
        request_timeout=or_cfg.get('request_timeout', 15),
    )
    exp_cfg = dict(cfg.get('experiments', {}) or {})
    if mechanism_overrides:
        exp_cfg['mechanisms'] = {**(exp_cfg.get('mechanisms', {}) or {}), **mechanism_overrides}
    out_dir = _HERE / 'logs'
    chosen_game = game_name or (exp_cfg.get('games') or ['zork1'])[0]
    return Runner(
        game_name=chosen_game,
        out_dir=out_dir,
        llm_client=llm,
        config=exp_cfg,
        textquests_root=textquests_root,
    )


if __name__ == '__main__':
    game_arg = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = _load_config()
    runner = _build_runner(cfg, game_name=game_arg)
    runner.run_once()
