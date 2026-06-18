from inspect_scout import scanner, llm_scanner, Transcript

@scanner(messages='all')
def accepted_local_error_scan() -> llm_scanner:
    return llm_scanner(
        question=("Does this transcript show the agent repeatedly accepting a local error or looping on a failed action? "
                  "If yes, answer 'Yes' and provide a short explanation and step numbers."),
        answer='string'
    )
