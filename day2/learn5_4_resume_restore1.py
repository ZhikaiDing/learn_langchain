from typing import TypedDict, Annotated, Literal
import operator

from langgraph.types import Command, Send, interrupt, Interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

import time
import random

# A
# ├ B
# └ C

class State(TypedDict):
    results: Annotated[list, operator.add] = []

def nodeA(state: State):
    print("[debug] call A")
    return {"results": ["A"]}

def nodeB(state: State):
    wait_t = random.uniform(0.1, 1)
    time.sleep(wait_t)
    print(f"[debug] finish B sleep {wait_t}s")

    resume = interrupt({
        "Q": "judge node B: Y/N?", 
        "node":"B"
    })
    return {"results": [("B", resume)]}

def nodeC(state: State):
    wait_t = random.uniform(0.1, 1)
    time.sleep(wait_t)
    print(f"[debug] finish C sleep {wait_t}s")

    resume = interrupt({
        "Q": "judge node C: Y/N?", 
        "node":"C"
    })
    return {"results": [("C", resume)]}

graph_builder = StateGraph(State)
graph_builder.add_node("nodeA", nodeA)
graph_builder.add_node("nodeB", nodeB)
graph_builder.add_node("nodeC", nodeC)
graph_builder.add_edge(START, "nodeA")
graph_builder.add_edge("nodeA", "nodeB")
graph_builder.add_edge("nodeA", "nodeC")
graph_builder.add_edge("nodeB", END)
graph_builder.add_edge("nodeC", END)

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

    B_resumes = True
    C_resumes = False

    # B & C
    interrupt_list:list[Interrupt] = tmp_state["__interrupt__"]
    resume_dict = {}
    for it in interrupt_list:
        if it.value["node"] == "B":
            resume_dict[it.id] = B_resumes
        elif it.value["node"] == "C":
            resume_dict[it.id] = C_resumes
    print("[resume]", json.dumps(resume_dict, indent=2, ensure_ascii=False))

    start_t = time.time()
    state = graph.invoke(Command(resume=resume_dict), config=cfg)
    print(f"[total time 2] use {time.time()-start_t}s")
    print("[final state]", state)
