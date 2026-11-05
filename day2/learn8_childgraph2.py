from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

#%% child child graph
class SubC1State(TypedDict):
    tag:str = "child child: SubC1"
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
ckpt = MemorySaver()
subC1 = subC1_builder.compile(ckpt)

#%% child graph
class SubC2State(TypedDict):
    tag:str = "child:SubC2"
    question: str = ""
    outputs: Annotated[list, operator.add] = []
    count: Annotated[int, operator.add] = 0
    subC1_cfg: dict = {}

def c2_nodeA(state:SubC2State):
    return {"outputs": [f"Q: {state['question']}"]}

def c2_subC1Invoke(state:SubC2State):
    subC1_state = subC1.invoke(
        SubC1State(question=state['question'], tag="SubC1"),
        state['subC1_cfg']
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
subC2 = subC2_builder.compile(ckpt) # checkpointer 可以用同一个

#%% parent
class State(TypedDict):
    tag:str = "parent"
    question: str = ""
    results: list = [] # 子节点使用了 ckpt, 且是累加的, 父节点覆盖即可
    count: Annotated[int, operator.add] = 0

    subC1_cfg: dict = {} # 带 ckpt 的执行需要有 config
    subC2_cfg: dict = {}

def nodeA(state: State):
    return {
        "question": f"{state['question']} (think and answer)"
    }

def nodeB_subC2Invoke(state: State):
    subC2_state = subC2.invoke(
        {
            "question": state['question'],
            "subC1_cfg": state['subC1_cfg'],
            "tag": "SubC2"
        },
        state['subC2_cfg']
    )
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
graph = graph_builder.compile(ckpt)

import json
inputs = ["问题1", "问题2", "问题3"]
cfg = {"configurable":{"thread_id":"tid-001"}}
cfgSubC1 = {"configurable":{"thread_id":"tid-001-1"}}
cfgSubC2 = {"configurable":{"thread_id":"tid-001-2"}}
for question in inputs:
    state = {
        "tag": "parent",
        "question":question,
        "subC1_cfg": cfgSubC1,
        "subC2_cfg": cfgSubC2
    }
    state = graph.invoke(state, cfg)
    print("[state]", json.dumps(state, indent=2, ensure_ascii=False))

def show_state_snapshot(
        state_snapshot, 
        keys:str | list[str] = [
            "values", "config", "interrupts", "next"
        ], # "all"
        idx = ""
):
    tag = idx if isinstance(idx, str) else f" {idx}"
    print("[state_snapshot" + tag +"]")
    head = " "
    # print(head, "[debug]", print("[state_snapshot]", state_snapshot))
    d = {
        "type": type(state_snapshot).__name__,
        "values": state_snapshot.values,
        "next": state_snapshot.next,
        "config": state_snapshot.config,
        "metadata": state_snapshot.metadata,
        "created_at": state_snapshot.created_at,
        "parent_config": state_snapshot.parent_config,
        "tasks": state_snapshot.tasks,
        "interrupts": state_snapshot.interrupts,
    }
    if keys == "all":
        keys = list(d.keys())
    for k in keys:
        print(head, f"[{k}]", d[k])

graphs = [
    # {"name":"parent", "graph": graph, "config": cfg},
    {"name":"subC2", "graph": subC2, "config": cfgSubC2},
    # {"name":"subC1", "graph": subC1, "config": cfgSubC1},
]

for d in graphs:
    # state 数据结构和 graph 绑定, 因此 ckpt 快照需要使用对应的 graph
    state_hist = d["graph"].get_state_history(d["config"])
    print("="*50)
    i = 0
    for snap in state_hist:
        show_state_snapshot(snap, idx=i)
        i += 1
        print("-"*40)
