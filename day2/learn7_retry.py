from typing import TypedDict, Annotated, Literal
import operator
import random

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

RETRY_N = 2
TOTAL_COUNT = 0
WAY_COUNT = 0

class State(TypedDict):
    results: Annotated[list, operator.add] = []
    cur_A_tag: str = ""
    count_success: Annotated[int, operator.add] = 0
    count_fallback: Annotated[int, operator.add] = 0
    count_except: Annotated[int, operator.add] = 0

def nodeA(state:State, runtime: Runtime):
    global TOTAL_COUNT, WAY_COUNT
    TOTAL_COUNT += 1
    attempt = runtime.execution_info.node_attempt
    key = random.random()
    if key <= 0.33:
        WAY_COUNT += 1
        print(f"{TOTAL_COUNT} [debug {WAY_COUNT}] try:{attempt}, success\n")
        return {
            "results": [f"nodeA success (try:{attempt})"], 
            "cur_A_tag": "success",
            "count_success": 1
        }
    elif key <= 0.66:
        WAY_COUNT += 1
        print(f"{TOTAL_COUNT} [debug {WAY_COUNT}] try:{attempt}, fallback\n")
        return {"cur_A_tag": "fallback"}
    else:
        tag = "*"
        end = ""
        if attempt==RETRY_N:
            WAY_COUNT += 1
            tag = WAY_COUNT
            end = "\n"
        print(f"{TOTAL_COUNT} [debug {tag}] try:{attempt}, failed" + end)
        raise TimeoutError(f"Node A out of time! (try:{attempt})")

def fallback_node(state:State):
    return {"results": ["nodeA fallback"], "count_fallback": 1}

def except_handle(state:State)->Command[Literal["except_post_node"]]:
    return Command(
        update={
            "cur_A_tag": "failed", 
            "count_except": 1
        },
        goto="except_post_node"
    )

def except_post_node(state:State):
    return {"results": ["nodeA failed post"]}

graph_builder = StateGraph(State)
graph_builder.add_node(
    "nodeA", nodeA,
    retry_policy=RetryPolicy(
        max_attempts = RETRY_N,
        retry_on = TimeoutError
    ),
    error_handler=except_handle
)
graph_builder.add_node("fallback_node", fallback_node)
graph_builder.add_node("except_post_node", except_post_node)
graph_builder.add_edge(START, "nodeA")
graph_builder.add_conditional_edges(
    "nodeA",
    lambda state: state["cur_A_tag"],
    {
        "success": END,
        "fallback": "fallback_node",
    }
)
graph_builder.add_edge("fallback_node", END)
graph_builder.add_edge("except_post_node", END)

ckpt = MemorySaver()
graph = graph_builder.compile(checkpointer=ckpt)
graph_draw = graph.get_graph().draw_mermaid()

import json
RUN_N = 12
cfg = {"configurable": {"thread_id": "tid-001"}}
for i in range(RUN_N):
    state = graph.invoke({"cur_A_tag":""}, cfg)
    # print(f"[loop {i}]", json.dumps(state, indent=2, ensure_ascii=False))

print("="*50)
snaps = graph.get_state_history(cfg)
i = 0
for snap in snaps:
    print(f"[snap {i}]")
    print("  [next]", snap.next)
    print("  [values]", json.dumps(snap.values, indent=2, ensure_ascii=False))
    print("  [tasks]")
    j = 0
    for task in snap.tasks:
        print(f"    [task {j} error]", task.error)
        j += 1
    i += 1
    if "__start__" in snap.next:
        print("="*50)
    else:
        print("-"*40)

print(graph_draw)
