import uuid
import time
import json
import os
import sys
from pathlib import Path
from logger import JSONLogger
from harness import AcceptedLocalErrorDetector
from openrouter_client import OpenRouterClient

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEXTQUESTS_ROOT = REPO_ROOT / 'textquests'

for path in (REPO_ROOT, DEFAULT_TEXTQUESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def _resolve_textquests_root(config):
    configured = (
        os.getenv('TEXTQUESTS_ROOT')
        or (config.get('textquests_root') if isinstance(config, dict) else None)
        or DEFAULT_TEXTQUESTS_ROOT
    )
    return Path(configured).expanduser().resolve()


def _load_textquests_env(textquests_root):
    if str(textquests_root) not in sys.path:
        sys.path.insert(0, str(textquests_root))
    from src.textquests_env import TextQuestsEnv
    return TextQuestsEnv

class Runner:
    def __init__(self, game_name, out_dir, model_client, config, textquests_root):
        self.game_name = game_name
        self.out_dir = Path(out_dir)
        self.logger = JSONLogger(self.out_dir)
        self.detector = AcceptedLocalErrorDetector(k=config.get('accepted_local_error_k',3), window=config.get('info_seek_window',10))
        self.model_client = model_client
        self.max_steps = config.get('max_steps',500)
        self.textquests_root = Path(textquests_root).expanduser().resolve()
        self.TextQuestsEnv = None

    def _resolve_game_path(self):
        candidate = self.textquests_root / 'data' / 'jiminy-cricket' / self.game_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f'Could not find TextQuests game folder: {candidate}. Set TEXTQUESTS_ROOT to an external TextQuests checkout.'
        )

    def _make_history_str(self, history):
        return [f"OBS: {o['obs']}\nACT: {o['action']}\nREAS: {o.get('reasoning','')}" for o in history]

    def run_once(self, seed=0):
        run_id = uuid.uuid4().hex
        if self.TextQuestsEnv is None:
            self.TextQuestsEnv = _load_textquests_env(self.textquests_root)
        env = self.TextQuestsEnv(str(self._resolve_game_path()), seed=seed)

        history = []
        autosaves = {}
        # initial observation via reset
        observation, info = env.reset()
        progress = 0
        for step in range(self.max_steps):
            # build model input as history strings
            history_str = self._make_history_str(history)
            prompt = '\n'.join(history_str + [f"OBSERVE: {observation}"])
            resp = self.model_client.generate(prompt, max_tokens=128, reasoning_enabled=False)
            parsed = self.model_client.parse_completion(resp)
            content = parsed.get('content') if isinstance(parsed, dict) else str(parsed)
            # naive parse: last non-empty line is action
            action = 'look'
            if content:
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                if lines:
                    action = lines[-1]

            # execute action
            try:
                new_observation, reward, game_over, info = env.step(action)
            except Exception as e:
                # log and break on unexpected env error
                error_record = {'run_id': run_id, 'step': step, 'error': str(e)}
                self.logger.log_step(run_id, error_record)
                break

            # compute progress from info if present
            new_progress = info.get('game_progress', progress) if isinstance(info, dict) else progress

            record = {
                'run_id': run_id,
                'step': step,
                'obs': observation,
                'action': action,
                'reasoning': content,
                'progress': new_progress,
                'reward': reward,
                'timestamp': time.time()
            }
            self.logger.log_step(run_id, record)
            history.append({'obs': observation, 'action': action, 'reasoning': content, 'progress': new_progress})
            # add action, progress, and the model's reasoning to the detector
            self.detector.add(action, new_progress, content)

            # autosave: save checkpoint
            try:
                ckpt_id = env.save_checkpoint(observation, info)
                autosaves[step] = ckpt_id
            except Exception:
                pass

            # If repetition detected, verify ALE via environment check
            if self.detector.check_repetition():
                # env_checker uses the current env to validate the extracted belief
                def env_checker(belief):
                    try:
                        state_tuple = env.get_state()
                        # env.get_state() returns (env_state, custom_state)
                        if isinstance(state_tuple, tuple) and len(state_tuple) >= 2:
                            state = state_tuple[1]
                        else:
                            state = getattr(env, 'state', {}) or {}
                    except Exception:
                        state = getattr(env, 'state', {}) or {}

                    # belief formats from harness.extract_belief
                    if belief[0] == 'have_item':
                        item = belief[1].lower()
                        taken = {k.lower() for k in state.get('taken_dict', {}).keys()} if isinstance(state.get('taken_dict', {}), dict) else set()
                        if item in taken:
                            return (False, {'type': 'inventory', 'found': True})
                        # scan checkpoints and current observation for mentions
                        combined = (observation or '') + ' ' + ' '.join([ck.get('observation','') for ck in getattr(env, 'checkpoints', [])])
                        if item in combined.lower():
                            return (False, {'type': 'observation', 'found_in_history': True})
                        return (True, {'type': 'none', 'found': False})

                    if belief[0] == 'item_in_loc':
                        # belief may be ('item_in_loc', item, loc) or ('item_in_loc', None, loc)
                        item = belief[1].lower() if belief[1] else None
                        loc = belief[2].lower() if len(belief) > 2 and belief[2] else None
                        combined = (observation or '') + ' ' + ' '.join([ck.get('observation','') for ck in getattr(env, 'checkpoints', [])])
                        lowered = combined.lower()
                        if item and loc and (item in lowered and loc in lowered):
                            return (False, {'type': 'observation', 'found': True})
                        # if loc mentioned but item never associated with loc -> contradiction
                        if loc and loc in lowered and (not item or item not in lowered):
                            return (True, {'type': 'contradiction', 'reason': 'loc_mentioned_but_item_not'})
                        return (True, {'type': 'none', 'found': False})

                    if belief[0] == 'clause':
                        clause = belief[1].lower()
                        combined = (observation or '') + ' ' + ' '.join([ck.get('observation','') for ck in getattr(env, 'checkpoints', [])])
                        if clause in combined.lower():
                            return (False, {'type': 'clause', 'found': True})
                        return (True, {'type': 'none', 'found': False})

                    return (False, {'reason': 'unknown_belief_type'})

                is_ale, details = self.detector.check_ale(env_checker)
                if is_ale:
                    # ALE confirmed: log details and run intervention
                    self.logger.save_transcript(run_id, history)
                    trigger_record = {'run_id': run_id, 'step': step, 'trigger': 'accepted_local_error', 'ale_details': details}
                    self.logger.log_step(run_id, trigger_record)

                    # Intervention: info-seeking using model's reasoning mode (as before)
                    window = 10
                    recent = self._make_history_str(history[-window:])
                    info_prompt = ("Summarize the recent history, list 3 hypotheses why the agent is stuck, and propose one concrete next game action.\n\n" + '\n'.join(recent))
                    info_resp = self.model_client.generate(info_prompt, max_tokens=256, reasoning_enabled=True)
                    info_parsed = self.model_client.parse_completion(info_resp)
                    mech_record = {'run_id': run_id, 'step': step, 'mechanism': 'info_seeking', 'output': info_parsed}
                    self.logger.log_step(run_id, mech_record)

                    # apply proposed action if parseable
                    proposed = ''
                    if isinstance(info_parsed, dict):
                        prop_text = info_parsed.get('content','')
                        lines = [l.strip() for l in prop_text.splitlines() if l.strip()]
                        if lines:
                            proposed = lines[-1]
                    else:
                        prop_text = str(info_parsed)
                        lines = [l.strip() for l in prop_text.splitlines() if l.strip()]
                        if lines:
                            proposed = lines[-1]

                    if proposed:
                        try:
                            obs2, reward2, game_over2, info2 = env.step(proposed)
                            after_record = {
                                'run_id': run_id,
                                'step': float(step) + 0.1,
                                'obs': obs2,
                                'action': proposed,
                                'reasoning': info_parsed,
                                'progress': info2.get('game_progress', new_progress) if isinstance(info2, dict) else new_progress,
                                'reward': reward2,
                                'timestamp': time.time()
                            }
                            self.logger.log_step(run_id, after_record)
                        except Exception as e:
                            self.logger.log_step(run_id, {'run_id': run_id, 'step': step, 'error': f'Intervention failed: {e}'})
                    break
                else:
                    # repetition but no ALE (or not yet verifiable)
                    self.logger.log_step(run_id, {'run_id': run_id, 'step': step, 'trigger': 'repetition_no_ale', 'details': details})

            if game_over:
                break

            # advance observation for next turn
            observation = new_observation
            progress = new_progress

        return run_id

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent / 'config.yaml'))
    textquests_root = _resolve_textquests_root(cfg)
    openrouter_cfg = cfg.get('openrouter', {}) or {}
    api_key = (
        openrouter_cfg.get('api_key')
        or openrouter_cfg.get('OPENROUTER_API_KEY')
        or os.getenv('OPENROUTER_API_KEY')
    )
    orc = OpenRouterClient(
        api_key=api_key,
        model=openrouter_cfg.get('model'),
        base_url=openrouter_cfg.get('base_url'),
    )
    r = Runner('zork1', Path(__file__).resolve().parent / 'logs', orc, cfg['experiments'], textquests_root)
    r.run_once()
