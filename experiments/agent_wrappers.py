from typing import Callable, Dict, Any

class BaselineAgent:
    def __init__(self, policy_fn: Callable):
        self.policy_fn = policy_fn

    async def act(self, history):
        return await self.policy_fn(history)

class InfoSeekingWrapper:
    def __init__(self, base_agent, model_client, window=10):
        self.base_agent = base_agent
        self.model_client = model_client
        self.window = window

    async def act(self, history):
        # call base agent to get its intended action
        reasoning, action = await self.base_agent.act(history)
        # if base is repeatedly failing, run info-seeking
        # here we always run info-seeking when wrapper invoked for demo; detector should gate it
        prompt = (
            "You are an assistant that helps an agent playing a text-based game.\n"
            "Given the recent history (most recent messages last), summarize the last {w} turns, list possible hypotheses why previous actions failed, and propose a single next action.\n\n"
            "History:\n".format(w=self.window)
        )
        prompt += '\n'.join(history[-self.window:])
        resp = self.model_client.generate(prompt, max_tokens=256)
        parsed = self.model_client.parse_completion(resp)
        # naive parsing: split lines, last line is action
        lines = parsed.strip().splitlines()
        if lines:
            proposed = lines[-1].strip()
            # return both reasoning and chosen action
            return parsed, proposed
        return parsed, action
