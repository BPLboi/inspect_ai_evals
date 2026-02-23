from inspect_ai import eval
# from debate import 

if __name__ == "__main__":
    eval("debate.py", model_roles = {
        "for_debater" : "openrouter/anthropic/claude-3.5-haiku",
        "against_debater" : "openrouter/gpt-4o-mini",
        "judge": "openrouter/anthropic/claude-sonnet-4.6"
    })

    # print(f"The debate winners were: {state.output.completion}")
    