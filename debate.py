"""
Simulates debates between two given AI agents, which are judged by a third AI.


TODO:
- Need some function to evaluate the winner of the AI debate
    - Need to assign a score to each model, somehow.

TODO (but not part of MVP):
- Control for whether each AI is forced to argue FOR or AGAINST
- Allow the judge to give qualitative answers, not just "debater a" or "debater b"
- Let models know how many turns they'll have left
"""

import re

from inspect_ai.agent import Agent, AgentState, agent, run
from inspect_ai.model import get_model

from inspect_ai import Task, task

from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    AnswerPattern,
    Score,
    Target,
    accuracy,
    stderr,
    scorer,
)

from inspect_ai.solver import (
    TaskState
)

@agent
def debate_agent(rounds=3) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        topic = state.messages[-1].content  # the debate prompt
        
        debater_a = get_model(role="for_debater")
        debater_b = get_model(role="against_debater")
        
        transcript = []
        for round in range(rounds):  # N rounds of debate
            # Debater A argues
            a_response = await debater_a.generate(
                f"Debate topic: {topic}\n\nPrior exchanges:\n{transcript}\n\nArgue FOR the proposition."
            )
            transcript.append(f"Debater A: {a_response.message.text}")
            
            # Debater B argues
            b_response = await debater_b.generate(
                f"Debate topic: {topic}\n\nPrior exchanges:\n{transcript}\n\nArgue AGAINST the proposition."
            )
            transcript.append(f"Debater B: {b_response.message.text}")
        
        # Judge decides
        judge_model = get_model(role="judge")
        verdict = await judge_model.generate(
            f"You are a debate judge. Here is the full debate:\n\n{chr(10).join(transcript)}\n\nWho won and why? Answer: 'Debater A' or 'Debater B' (without the quotes)."
        )

        state.messages.append(verdict.message)
        state.output = verdict
        return state
    return execute

def get_dataset() -> list[Sample]:
    dataset = [
        Sample(
            input="Cereal is a soup."
        )
        # Sample(
        #     input="You should fight one horse-sized duck instead of 100 duck-sized horses."
        # ),
        # Sample(
        #     input="A hot dog is a sandwich."
        # ),
        # Sample(
        #     input="Cats are secretly smarter than humans."
        # ),
        # Sample(
        #     input="Pineapple belongs on pizza."
        # ),
        # Sample(
        #     input="It is morally wrong to eat the last slice of pizza without asking."
        # ),
        # Sample(
        #     input="The world would be better if everyone had personal theme music."
        # ),
        # Sample(
        #     input="Socks with sandals are a legitimate fashion choice."
        # ),
        # Sample(
        #     input="A burrito is just a folded taco."
        # ),
        # Sample(
        #     input="Naps should be mandatory for adults."
        # ),
        # Sample(
        #     input="Water is wet."
        # ),
        # Sample(
        #     input="If you replace every part of a broom over time, it is still the same broom."
        # ),
        # Sample(
        #     input="It would be better to sing everything instead of speaking."
        # ),
        # Sample(
        #     input="Ghosts are just misunderstood roommates."
        # ),
        # Sample(
        #     input="Clapping should be replaced with jazz hands."
        # ),
        # Sample(
        #     input="A pop-tart is a type of ravioli."
        # ),
        # Sample(
        #     input="It is acceptable to recline your seat on an airplane."
        # ),
        # Sample(
        #     input="Aliens are avoiding Earth because of our reality TV shows."
        # ),
        # Sample(
        #     input="If animals could talk, cats would be the rudest species."
        # ),
        # Sample(
        #     input="Spaghetti for hair would be better than maple syrup for sweat."
        # )
    ]

    return dataset


@task
def debate(shuffle=False):
    return Task(
        dataset=get_dataset(),
        solver=debate_agent()
    )