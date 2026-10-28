from pydantic import BaseModel
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableLambda
import operator

import random
import time

class State(BaseModel):
    callworker_mode: str = "send id" # "send id" , "send", "str"
    cur_worker_num: int = 0
    cur_workouts: Annotated[list, operator.add] = []
    cur_choice: str = ""
    cur_output: float = 0
    count: Annotated[int, operator.add] = 0

class WorkerState(BaseModel):
    worker_id: int = 0

def worker(state: WorkerState | State): # worker 接收 局部专属state    
    try:
        worker_id = state.worker_id
        print("[debug] worker id ok")
    except:
        worker_id = state.count # 由于是并行调用，这里的 worker_id 会有问题
        print("[info] unk worker id, use count")

    sleep_t = random.uniform(0.1, 1.0)
    print("[debug] worker", worker_id, "start")
    time.sleep(sleep_t)
    print("[debug] worker", worker_id, "finish, sleep", round(sleep_t,2), "s")
    return { # 修改仍针对 全局state
        "cur_workouts": [
            {
                "result": random.random(),
                "worker_id": worker_id
            }
        ],
        "count": 1
    }

def call_workers(state:State) -> Command[Literal["worker"]]:
    # tips: Command 实现动态分支数
    if state.callworker_mode == "send id":
        # Send 列表, 局部 state, 结果顺序和 Send 列表一致
        sends = [
            Send("worker", WorkerState(worker_id=i)) \
                for i in range(state.cur_worker_num) # cur_worker_num 是变化的
        ]
    elif state.callworker_mode == "send":
        # Send 列表, 传全局 state, 结果顺序和 Send 列表一致
        sends = [
            Send("worker", state) \
                for i in range(state.cur_worker_num)
        ]
    else:
        # 普通列表: 去重, 只调用 1次 worker
        sends = [
            "worker" for i in range(state.cur_worker_num)
        ]
        print("[debug] str list:", sends)

    return Command(goto=sends) # tips: 并行调度

def sumup(state:State):
    ans = 0
    for d in state.cur_workouts:
        ans += d["result"]
    return {"cur_output": ans}

#%% build & compile
graph_builder = StateGraph(State)

graph_builder.add_node("call_workers", call_workers)
graph_builder.add_node("worker", worker)
graph_builder.add_node("sumup", sumup)

graph_builder.add_edge(START, "call_workers")
# 不能写: graph_builder.add_edge("call_workers", "worker")
#    否则多一个连接
graph_builder.add_edge("worker", "sumup")
graph_builder.add_edge("sumup", END)

graph = graph_builder.compile()
print(graph.get_graph().draw_mermaid())
print("="*50)

#%% test
import json
callworker_mode = "send id" # "send id" , "send", "str"
worker_nums = [1, 5, 3, 2]
for worker_num in worker_nums:
    state = {
        "callworker_mode": callworker_mode,
        "cur_worker_num": worker_num,
    } # 重置 state
    state = graph.invoke(state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print("-"*40)
