"""LLM client and response utilities.

OpenRouterClient wraps the OpenRouter chat-completions API with automatic
retries. extract_action_and_reasoning parses the JSON {action, reasoning}
format that game agents are prompted to return.

Typical usage
-------------
from llm import OpenRouterClient, extract_action_and_reasoning

client = OpenRouterClient(api_key=..., model='google/gemma-4-26b-a4b-it')
resp   = client.generate('What is 2+2?', max_tokens=64)
parsed = client.parse_completion(resp)
action, reasoning = extract_action_and_reasoning(parsed['content'])
"""

import json
import os
import re
import time

import requests


def _normalize_base_url(base_url):
    base_url = (base_url or '').strip().rstrip('/')
    if not base_url:
        return 'https://openrouter.ai/api/v1'
    if base_url.endswith('/api'):
        return base_url + '/v1'
    return base_url


class OpenRouterClient:
    """HTTP client for the OpenRouter chat-completions endpoint."""

    def __init__(self, api_key=None, model=None, base_url=None, request_timeout=15):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise RuntimeError(
                'OPENROUTER_API_KEY not set. '
                'Export it in your shell or add it to config.yaml under openrouter.api_key.'
            )
        self.model = model or os.getenv('OPENROUTER_MODEL') or 'openrouter/free'
        self.request_timeout = request_timeout
        self.base_url = _normalize_base_url(
            base_url or os.getenv('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1'
        )
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        })

    def generate(self, prompt, temperature=0.2, max_tokens=512, timeout=None, reasoning_enabled=False):
        """Send a prompt and return the raw API response dict.

        Args:
            prompt: str (single user message) or list of {role, content} dicts.
            reasoning_enabled: pass True to enable chain-of-thought on supported models.
        """
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if reasoning_enabled:
            payload['reasoning'] = {'enabled': True}

        url = self.base_url.rstrip('/') + '/chat/completions'
        effective_timeout = timeout if timeout is not None else self.request_timeout
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, timeout=effective_timeout)
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt < 2:
                    time.sleep(1 + attempt)
                else:
                    raise

    def parse_completion(self, resp):
        """Extract content and reasoning_details from a raw API response.

        Returns a dict with keys: content (str), reasoning_details, raw.
        """
        if not resp:
            return {'content': '', 'reasoning_details': None, 'raw': resp}
        if isinstance(resp, dict):
            usage = resp.get('usage') or {}
            tokens = usage.get('total_tokens') or (
                (usage.get('prompt_tokens') or 0) + (usage.get('completion_tokens') or 0)
            )
            choices = resp.get('choices') or []
            if choices:
                msg = choices[0].get('message') or {}
                return {
                    'content': msg.get('content') or choices[0].get('text') or '',
                    'reasoning_details': msg.get('reasoning_details'),
                    'tokens': tokens,
                    'raw': resp,
                }
            if 'output' in resp:
                return {'content': str(resp['output']), 'reasoning_details': None, 'tokens': tokens, 'raw': resp}
        return {'content': str(resp), 'reasoning_details': None, 'tokens': 0, 'raw': resp}


def extract_action_and_reasoning(content, default_action='look'):
    """Parse a model response into (action, reasoning).

    Expects a JSON object with keys "action" and "reasoning". Falls back to
    using the last non-empty line as the action if JSON parsing fails.
    """
    text = (content or '').strip()
    if not text:
        return default_action, ''

    # collect JSON candidates in order of preference
    candidates = []
    fenced = re.search(r'```json\s*(.*?)\s*```', text, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if text.startswith('{') and text.endswith('}'):
        candidates.append(text)
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            action = str(parsed.get('action', '')).strip()
            reasoning = str(parsed.get('reasoning', '')).strip()
            if action:
                return action, reasoning or text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1], text) if lines else (default_action, text)


_VALID_BELIEF_TYPES = ('have_item', 'item_in_loc', 'none')


def _coerce_belief(raw):
    """Normalize a self-reported belief dict; return None if unusable."""
    if not isinstance(raw, dict):
        return None
    btype = str(raw.get('type', 'none')).strip().lower()
    if btype not in _VALID_BELIEF_TYPES or btype == 'none':
        return None
    item = str(raw.get('item') or '').strip() or None
    location = str(raw.get('location') or '').strip() or None
    if btype == 'have_item' and not item:
        return None
    if btype == 'item_in_loc' and not location:
        return None
    return {'type': btype, 'item': item, 'location': location}


def extract_step_fields(content, default_action='look'):
    """Parse a model response into the full structured step schema.

    Superset of extract_action_and_reasoning: also pulls goal, subgoal,
    belief, and confidence when the model reports them. Any field the model
    omits or reports malformed is simply absent from the result (None /
    empty) rather than reconstructed via a text-parsing fallback — mechanisms
    that depend on a field just get no signal that step.

    Returns a dict with keys: action, reasoning, goal, subgoal, belief
    (None or {type, item, location}), confidence (float or None).
    """
    text = (content or '').strip()
    result = {
        'action': default_action,
        'reasoning': '',
        'goal': None,
        'subgoal': None,
        'belief': None,
        'confidence': None,
    }
    if not text:
        return result

    candidates = []
    fenced = re.search(r'```json\s*(.*?)\s*```', text, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if text.startswith('{') and text.endswith('}'):
        candidates.append(text)
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        action = str(parsed.get('action', '')).strip()
        if not action:
            continue
        result['action'] = action
        result['reasoning'] = str(parsed.get('reasoning', '')).strip() or text
        result['goal'] = str(parsed.get('goal') or '').strip() or None
        result['subgoal'] = str(parsed.get('subgoal') or '').strip() or None
        result['belief'] = _coerce_belief(parsed.get('belief'))
        try:
            confidence = parsed.get('confidence')
            result['confidence'] = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            result['confidence'] = None
        return result

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result['action'] = lines[-1] if lines else default_action
    result['reasoning'] = text
    return result
