from inspect_ai import eval
# from debate import 

if __name__ == "__main__":
    eval("debate.py", model_roles = {
        "for_debater" : "anthropic/claude-3.5-haiku",
        "against_debater" : "gpt-4o-mini",
        "grader": "anthropic/claude-sonnet-4.6"
    })

    # print(f"The debate winners were: {state.output.completion}")
    