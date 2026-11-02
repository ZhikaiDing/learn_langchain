from typing import TypedDict, Annotated, Literal
import operator

from langgraph.types import Command, Send, interrupt, Interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

import time
import random

# A
# ├ B
# │ └ D0,D1,D2
# └ C
#   └ E0,E1,E2

class State(TypedDict):
    results: Annotated[list, operator.add] = []

class StateD(TypedDict):
    from_node: str = ""
    idx: int = 0

class StateE(TypedDict):
    idx: int = 0

def nodeA(state: State):
    print("[debug] call A")
    return {"results": ["A"]}

def nodeB(state: State) -> Command[Literal["nodeD"]]:
    print("[debug] call B")
    sends = [
        Send("nodeD", {"from_node":"B", "idx":i}) \
            for i in range(3)
    ]
    return Command(goto=sends, update={"results": ["B"]})

def nodeD(d:StateD):
    wait_t = random.uniform(0.1, 1)
    time.sleep(wait_t)
    print(f"[debug] finish D{d['idx']} sleep {wait_t}s")

    resume = interrupt({
        "Q": "judge node D: Y/N?", 
        "from":d["from_node"], 
        "node":"D", 
        "idx":d["idx"]
    })
    return {"results": [
        (f"D{d['idx']}", resume)
    ]}

def nodeC(state: State) -> Command[Literal["nodeE"]]:
    print("[debug] call C")
    sends = [
        Send("nodeE", {"idx":i}) for i in range(3)
    ]
    return Command(goto=sends, update={"results": ["C"]})

def nodeE(d:StateE):
    wait_t = random.uniform(0.1, 1)
    time.sleep(wait_t)
    print(f"[debug] finish E{d['idx']} sleep {wait_t}s")

    resume = interrupt({
        "Q": "judge node E: Y/N?", 
        "node":"E", 
        "idx":d["idx"]
    })
    return {"results": [(f"E{d['idx']}", resume)]}

graph_builder = StateGraph(State)
graph_builder.add_node("nodeA", nodeA)
graph_builder.add_node("nodeB", nodeB)
graph_builder.add_node("nodeC", nodeC)
graph_builder.add_node("nodeD", nodeD)
graph_builder.add_node("nodeE", nodeE)
graph_builder.add_edge(START, "nodeA")
graph_builder.add_edge("nodeA", "nodeB")
graph_builder.add_edge("nodeA", "nodeC")
graph_builder.add_edge("nodeD", END)
graph_builder.add_edge("nodeE", END)

ckpt = MemorySaver()
graph = graph_builder.compile(checkpointer=ckpt)
graph_pic_str = graph.get_graph().draw_mermaid()
print(graph_pic_str, "\n"+"="*50)

if __name__ == "__main__":
    print("start running...")

    import json

    start_t = time.time()
    cfg = {"configurable": {"thread_id": "tid-001"}}
    tmp_state = graph.invoke({"results":[]}, config=cfg)
    # print("[debug tmp_state]", tmp_state)
    print(f"[total time 1] use {time.time()-start_t}s")

    D_resumes = [True, False, True]
    E_resumes = [False, True, True]

    # D & E
    interrupt_list:list[Interrupt] = tmp_state["__interrupt__"]
    resume_dict = {}
    for it in interrupt_list:
        if it.value["node"] == "D":
            resume_dict[it.id] = D_resumes[it.value["idx"]]
        elif it.value["node"] == "E":
            resume_dict[it.id] = E_resumes[it.value["idx"]]
    print("[resume]", json.dumps(resume_dict, indent=2, ensure_ascii=False))

    start_t = time.time()
    state = graph.invoke(Command(resume=resume_dict), config=cfg)
    print(f"[total time 2] use {time.time()-start_t}s")
    print("[final state]", state)
