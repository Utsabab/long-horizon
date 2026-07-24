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
from channels import AcceptedLocalErrorDetector, DriftDetector, make_env_checker
from env import find_game_path, load_env_class, resolve_root
from llm import OpenRouterClient, extract_action_and_reasoning
from logger import RunLogger
from mechanisms import InfoSeekingMechanism, MemoryMechanism


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
        self.memory_enabled        = mem_cfg.get('enabled', True)
        self.memory_max_visited    = mem_cfg.get('max_visited', 25)
        self.memory_max_discoveries = mem_cfg.get('max_discoveries', 10)

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

    def _run_loop(self, env, logger, run_id, history, observation, info, progress, score,
                  start_step, detector, env_checker, info_seeking=None, capture_fork=False,
                  drift_detector=None, memory=None):
        """Run steps [start_step, self.max_steps) on a live env.

        Checks for an Accepted-Local-Error before deciding each step's action
        (using the trailing action/progress/reasoning window built up by prior
        steps). When confirmed:
          - always logged via logger.add_ale()
          - if info_seeking is given, its intervene() result replaces the
            normal LLM-chosen action for this step (the "harness" branch)
          - if capture_fork is True, the checkpoint saved at the end of the
            *previous* step (i.e. the state right before this decision) and a
            snapshot of steps/milestones logged so far are captured once, on
            the first occurrence, so a caller can later restore to that exact
            decision point and re-run with a different policy.

        Returns a dict: observation, progress, score, game_over, last_step,
        history, fork (None unless capture_fork captured one).
        """
        game_over = False
        last_checkpoint_id = None
        fork = None
        step = start_step - 1

        for step in range(start_step, self.max_steps):
            is_ale, details = detector.check_ale(env_checker)
            used_intervention = False

            if is_ale:
                logger.add_ale({'step': step, **details})
                if capture_fork and fork is None:
                    fork = {
                        'step': step,
                        'checkpoint_id': last_checkpoint_id,
                        'progress': progress,
                        'score': score,
                        'steps_snapshot': list(logger._steps),
                        'milestones_snapshot': list(logger._milestones),
                    }
                if info_seeking is not None:
                    try:
                        proposed_action, parsed = info_seeking.intervene(
                            format_history(history), details['belief'], details['evidence'],
                        )
                    except Exception as exc:
                        logger.add_error({'step': step, 'error': f'info_seeking.intervene failed: {exc}'})
                        proposed_action = ''
                    if proposed_action:
                        action = proposed_action
                        reasoning = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
                        tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0
                        used_intervention = True

            if not used_intervention:
                scratchpad = memory.render() if memory else None
                messages = build_step_messages(self.game_name, history, observation,
                                               scratchpad=scratchpad)
                try:
                    resp = self.llm.generate(messages, max_tokens=128, reasoning_enabled=False)
                    parsed = self.llm.parse_completion(resp)
                except Exception as exc:
                    logger.add_error({'step': step, 'error': f'llm.generate failed: {exc}'})
                    break
                content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
                tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0
                action, reasoning = extract_action_and_reasoning(content)

            try:
                new_observation, reward, game_over, info = env.step(action)
            except Exception as exc:
                logger.add_error({'step': step, 'error': str(exc)})
                break

            new_progress = info.get('game_progress', progress) if isinstance(info, dict) else progress
            score = info.get('score', 0) if isinstance(info, dict) else 0

            logger.add_step({
                'step': step,
                'obs': observation,
                'action': action,
                'reasoning': reasoning,
                'progress': new_progress,
                'score': score,
                'reward': reward,
                'tokens': tokens,
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

            # --- M2: update memory scratchpad ---
            if memory:
                memory.update(observation, new_observation, action, reasoning)

            # --- Drift detection ---
            if drift_detector:
                drift_detector.add(action, new_progress, reasoning)
                is_drifting, drift_details = drift_detector.check_drift()
                if is_drifting:
                    logger.add_drift_event({'step': step, **drift_details})
                    print(f'[{run_id[:8]}] step {step:>4}  DRIFT  score={drift_details["drift_score"]:.2f}'
                          f'  (stagnation={drift_details["stagnation"]:.2f}'
                          f'  loop={drift_details["loop_rate"]:.2f}'
                          f'  incoherence={drift_details["incoherence"]:.2f})')

            try:
                last_checkpoint_id = env.save_checkpoint(new_observation, info)
            except Exception:
                pass

            detector.add(action, new_progress, reasoning)

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

    def run_once(self, seed=0):
        """Run one episode with plain ALE + drift detection/logging, no intervention.

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

        # --- Drift detector (composite: stagnation + loop + coherence) ---
        drift_detector = None
        if self.drift_enabled:
            drift_detector = DriftDetector(
                window=self.drift_window,
                threshold=self.drift_threshold,
                w_stagnation=self.drift_w_stagnation,
                w_loop=self.drift_w_loop,
                w_coherence=self.drift_w_coherence,
            )
            print(f'[{run_id[:8]}] drift detector enabled  '
                  f'(window={self.drift_window}, threshold={self.drift_threshold:.2f})')

        # --- Memory scratchpad (M2) ---
        memory = None
        if self.memory_enabled:
            memory = MemoryMechanism(max_visited=self.memory_max_visited,
                                     max_discoveries=self.memory_max_discoveries)
            # Seed location from initial observation
            memory.update('', observation, '', '')

        result = self._run_loop(
            env, self.logger, run_id, [], observation, info, 0, 0, 0,
            detector, env_checker, info_seeking=None, capture_fork=False,
            drift_detector=drift_detector, memory=memory,
        )

        path = self.logger.save(run_id, self.game_name, {
            'total_steps': result['last_step'] + 1,
            'final_progress': result['progress'],
            'final_score': result['score'],
            'game_over': result['game_over'],
            'label': 'baseline',
            'fork_step': None,
        })
        print(f'[{run_id[:8]}] done — {result["last_step"] + 1} steps, progress {result["progress"]:.1f}%  →  {path}')

        return run_id

    def run_comparison(self, seed=0):
        """Run a baseline episode, and if an ALE is detected, fork a harness episode.

        The baseline (control) plays the whole episode with no intervention. If
        an Accepted-Local-Error is confirmed at some step, the env is rewound
        (via restore_checkpoint) to the decision point right before that step,
        and a second episode (harness/treatment) is played from there with
        InfoSeekingMechanism intervening whenever an ALE is confirmed, instead
        of letting the model act on the flagged belief unchecked.

        Returns
        -------
        (baseline_run_id, harness_run_id_or_None, fork_step_or_None)
        """
        baseline_id = uuid.uuid4().hex
        env = self._new_env(seed)
        observation, info = env.reset()
        detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        env_checker = make_env_checker(env)

        result = self._run_loop(
            env, self.logger, baseline_id, [], observation, info, 0, 0, 0,
            detector, env_checker, info_seeking=None, capture_fork=True,
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
        for a, p, r in zip(detector.actions, detector.progress, detector.reasonings):
            harness_detector.add(a, p, r)

        info_seeking = InfoSeekingMechanism(self.llm, window=self.ale_window)

        for step_record in fork['steps_snapshot']:
            self.logger.add_step(step_record)
        for milestone_record in fork['milestones_snapshot']:
            self.logger.add_milestone(milestone_record)

        harness_id = uuid.uuid4().hex
        harness_history = result['history'][:fork['step']]

        harness_result = self._run_loop(
            env, self.logger, harness_id, harness_history, obs, ckpt_info,
            fork['progress'], fork['score'], fork['step'],
            harness_detector, env_checker, info_seeking=info_seeking, capture_fork=False,
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


def _build_runner(cfg, game_name=None):
    """Construct a Runner from a config dict."""
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
    exp_cfg = cfg.get('experiments', {})
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
