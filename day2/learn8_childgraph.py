from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

#%% child child graph
class SubC1State(TypedDict):
    question: str = ""
    output: str = ""
    count: Annotated[int, operator.add] = 0

def c1_nodeA(state:SubC1State):
    return {
        "output": f"A: answer of {state['question']}",
        "count": 1
    }

subC1_builder = StateGraph(SubC1State)
subC1_builder.add_node("c1_nodeA", c1_nodeA)
subC1_builder.add_edge(START, "c1_nodeA")
subC1_builder.add_edge("c1_nodeA", END)
subC1 = subC1_builder.compile()

#%% child graph
class SubC2State(TypedDict):
    question: str = ""
    outputs: Annotated[list, operator.add] = []
    count: Annotated[int, operator.add] = 0

def c2_nodeA(state:SubC2State):
    return {"outputs": [f"Q: {state['question']}"]}

def c2_subC1Invoke(state:SubC2State):
    subC1_state = subC1.invoke(
        SubC1State(question=state['question'])
    )
    return {
        "outputs": [subC1_state["output"]],
        "count": 1
    }

subC2_builder = StateGraph(SubC2State)
subC2_builder.add_node("c2_nodeA", c2_nodeA)
subC2_builder.add_node("subC1", c2_subC1Invoke)
subC2_builder.add_edge(START, "c2_nodeA")
subC2_builder.add_edge("c2_nodeA", "subC1")
subC2_builder.add_edge("subC1", END)
subC2 = subC2_builder.compile()

#%% parent
class State(TypedDict):
    question: str = ""
    results: Annotated[list, operator.add] = []
    count: Annotated[int, operator.add] = 0

def nodeA(state: State):
    return {
        "question": f"{state['question']} (think and answer)"
    }

def nodeB_subC2Invoke(state: State):
    subC2_state = subC2.invoke({"question": state['question']})
    return {
        "results": subC2_state["outputs"],
        "count": 1
    }

graph_builder = StateGraph(State)
graph_builder.add_node("nodeA", nodeA)
graph_builder.add_node("subC2", nodeB_subC2Invoke)
graph_builder.add_edge(START, "nodeA")
graph_builder.add_edge("nodeA", "subC2")
graph_builder.add_edge("subC2", END)
graph = graph_builder.compile()

import json
inputs = ["问题1", "问题2", "问题3"]
state = {}
for question in inputs:
    state["question"] = question
    state = graph.invoke(state)
    print("[state]", json.dumps(state, indent=2, ensure_ascii=False))
