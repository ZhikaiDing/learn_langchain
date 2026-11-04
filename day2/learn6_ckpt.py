from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import StateSnapshot, interrupt, Command
from langgraph.graph.state import CompiledStateGraph
from copy import deepcopy

class State(TypedDict):
    results: Annotated[list, operator.add] = []
    count: Annotated[int, operator.add] = 0
    countA: Annotated[int, operator.add] = 0

def nodeA(state:State):
    return {
        "results": [f"A{state['count']}"],
        "countA" : 1
    }

def nodeB(state:State):
    resume = interrupt("use B? (Y/N)")
    if resume:
        return {"results": [f"B{state['count']}"]}
    else:
        return

graph_builder = StateGraph(State)
graph_builder.add_node("nodeA", nodeA)
graph_builder.add_node("nodeB", nodeB)
graph_builder.add_edge(START, "nodeA")
graph_builder.add_edge("nodeA", "nodeB")
graph_builder.add_edge("nodeB", END)

ckpt = MemorySaver()
graph = graph_builder.compile(checkpointer=ckpt)

def show_state_snapshot(
        state_snapshot:StateSnapshot, 
        keys:str | list[str] = [
            "values", "config", "interrupts", "next"
        ], # "all"
):
    print("[state_snapshot]")
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

def find_state(
        graph:CompiledStateGraph, state_snapshot:StateSnapshot,
        config
):
    cfg = state_snapshot.config
    state_snapshots = graph.get_state_history(config)
    matched_id = -1
    i = 0
    for snap in state_snapshots:
        matched = True
        k0 = "configurable"
        for k in cfg[k0]:
            if cfg[k0][k] != snap.config[k0][k]:
                matched = False
                break
        if matched:
            matched_id = i
            print("[debug] keys:", list(cfg[k0].keys()))
            print("   hist keys:", list(snap.config[k0].keys()))
        i += 1
    print(f"[snap match] id:{matched_id} in {i} states")

NEED_SHOW_ALL_HIST = True
cfg = {"configurable": {"thread_id": "tid-001"}}
agree_B = [True, False]
for i in range(len(agree_B)):
    state = graph.invoke({"count":1}, config=cfg)

    print("\n===" + f" loop {i} " + "="*40)
    print("\n---" + " B stop " + "-"*30)
    # 1. 初始运行会被 B 结点的 interrupt 打断
    print("[tmp state]", state)
    state_snapshot = graph.get_state(cfg) # 查找最新
    show_state_snapshot(state_snapshot)
    find_state(graph, state_snapshot, cfg) # 校验
    ckpt_id = state_snapshot.config["configurable"]["checkpoint_id"]
    # 注: 当 cfg 只包含 thread_id, 查找到的 snap 是最新的

    print("\n---" + " B resume " + "-"*30)
    # 2. 给 interrupt 传入人工判断，恢复图的运行
    state = graph.invoke(Command(resume=agree_B[i]), config=cfg)
    print("[state]", state)
    state_snapshot = graph.get_state(cfg)
    show_state_snapshot(state_snapshot)
    find_state(graph, state_snapshot, cfg)

    print("\n---" + " update state " + "-"*30)
    # 3. 手动更新快照
    graph.update_state(cfg, {"countA": -state_snapshot.values["countA"]})
    # 注意: countA 的更新是通过 operator.add 累加进去的
    # 这里通过传递 负值 来将其置0
    state_snapshot = graph.get_state(cfg)
    show_state_snapshot(state_snapshot)
    find_state(graph, state_snapshot, cfg)

    print("\n---" + " find B stop state " + "-"*30)
    # 4. 通过 thread_id + checkpoint_id 查询 ckpt 快照
    #   thread_id 是必须的
    #   额外写 checkpoint_id 可以定位到 历史中的某个快照
    tmp_cfg = deepcopy(cfg)
    tmp_cfg["configurable"]["checkpoint_id"] = ckpt_id
    state_snapshot = graph.get_state(tmp_cfg) # 查找
    show_state_snapshot(state_snapshot)
    find_state(graph, state_snapshot, cfg) # 校验

if NEED_SHOW_ALL_HIST:
    print("\n===" + f" state history " + "="*40)
    state_snapshots = graph.get_state_history(cfg)
    si = 0
    for state_snapshot in state_snapshots:
        print("---" + f" state {si}" + "-"*20)
        show_state_snapshot(state_snapshot)
        si += 1
