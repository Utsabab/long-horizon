from inspect_scout import scanner, llm_scanner


@scanner(messages='all')
def accepted_local_error_scan() -> llm_scanner:
    return llm_scanner(
        question=(
            "Classify whether this transcript shows Accepted-Local-Error behavior. Return JSON only, with no markdown or extra text, "
            "using exactly this schema: {\"verdict\": \"Yes\"|\"No\", \"is_accepted_local_error\": true|false, "
            "\"repeated_action\": string|null, \"repeated_steps\": [int, ...], \"progress_observed\": string, "
            "\"why\": string, \"confidence\": \"high\"|\"medium\"|\"low\"}. "
            "Answer Yes only if the agent repeats the same failed or nearly identical action 3 or more times in a short span, "
            "there is no meaningful progress or state change in the transcript, and the repetition is consistent with the agent "
            "persisting in a mistaken local belief. Answer No if the agent changes strategy, makes progress, or there is not enough evidence. "
            "When the answer is No, set is_accepted_local_error to false, repeated_action to null, repeated_steps to an empty list, "
            "and explain the main reason in why."
        ),
        answer='string'
    )