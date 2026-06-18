import os
import time
import requests

class OpenRouterClient:
    def __init__(self, api_key=None, model=None, base_url=None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.model = model or os.getenv("OPENROUTER_MODEL") or "openrouter/free"
        # default to documented openrouter host
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def generate(self, prompt, temperature=0.2, max_tokens=512, timeout=30, reasoning_enabled=False):
        # accept prompt as string or messages list
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if reasoning_enabled:
            payload["reasoning"] = {"enabled": True}

        url = self.base_url.rstrip('/') + '/chat/completions'
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < 2:
                    time.sleep(1 + attempt)
                else:
                    raise

    def parse_completion(self, resp):
        if not resp:
            return {"content": "", "reasoning_details": None, "raw": resp}
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices and isinstance(choices, list):
                first = choices[0]
                msg = first.get("message") or {}
                content = msg.get("content") or first.get("text") or ""
                reasoning_details = msg.get("reasoning_details") if isinstance(msg, dict) else None
                return {"content": content, "reasoning_details": reasoning_details, "raw": resp}
            if "output" in resp:
                return {"content": str(resp["output"]), "reasoning_details": None, "raw": resp}
        return {"content": str(resp), "reasoning_details": None, "raw": resp}
