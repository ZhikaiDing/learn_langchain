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
#   └ E
#     └ F0,F1,F2

class State(TypedDict):
    results: Annotated[list, operator.add] = []

class StateD(TypedDict):
    from_node: str = ""
    idx: int = 0

class StateF(TypedDict):
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

def nodeC(state: State):
    print("[debug] call C")
    return {"results": ["C"]}

def nodeE(state: State) -> Command[Literal["nodeF"]]:
    print("[debug] call E")
    sends = [
        Send("nodeF", {"idx":i}) for i in range(3)
    ]
    return Command(goto=sends, update={"results": ["E"]})

def nodeF(d:StateF):
    wait_t = random.uniform(0.1, 1)
    time.sleep(wait_t)
    print(f"[debug] finish F{d['idx']} sleep {wait_t}s")

    resume = interrupt({
        "Q": "judge node F: Y/N?", 
        "node":"F", 
        "idx":d["idx"]
    })
    return {"results": [(f"F{d['idx']}", resume)]}

graph_builder = StateGraph(State)
graph_builder.add_node("nodeA", nodeA)
graph_builder.add_node("nodeB", nodeB)
graph_builder.add_node("nodeC", nodeC)
graph_builder.add_node("nodeD", nodeD)
graph_builder.add_node("nodeE", nodeE)
graph_builder.add_node("nodeF", nodeF)
graph_builder.add_edge(START, "nodeA")
graph_builder.add_edge("nodeA", "nodeB")
graph_builder.add_edge("nodeA", "nodeC")
graph_builder.add_edge("nodeC", "nodeE")
graph_builder.add_edge("nodeD", END)
graph_builder.add_edge("nodeF", END)

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
    print(f"[total time 0] use {time.time()-start_t}s")

    D_resumes = [True, False, True]
    F_resumes = [False, True, True]

    def setup_resume_dict(tmp_state):
        interrupt_list:list[Interrupt] = tmp_state["__interrupt__"]
        resume_dict = {}
        for it in interrupt_list:
            if it.value["node"] == "D":
                resume_dict[it.id] = D_resumes[it.value["idx"]]
            elif it.value["node"] == "F":
                resume_dict[it.id] = F_resumes[it.value["idx"]]
        print("[resume]", json.dumps(resume_dict, indent=2, ensure_ascii=False))
        return resume_dict

    # D
    start_t = time.time()
    tmp_state = graph.invoke(
        Command(resume=setup_resume_dict(tmp_state)), 
        config=cfg
    )
    print(f"[total time D] use {time.time()-start_t}s")

    # F
    start_t = time.time()
    tmp_state = graph.invoke(
        Command(resume=setup_resume_dict(tmp_state)), 
        config=cfg
    )
    print(f"[total time F] use {time.time()-start_t}s")

    print("[final state]", tmp_state)
