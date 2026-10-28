from typing import TypedDict, Annotated, Literal
import operator
from pydantic import BaseModel
from langgraph.types import Command, Send, interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 主状态
class OverallState(TypedDict):
    queries: list[str]
    results: Annotated[list, operator.add]

# worker局部状态
class WorkerState(TypedDict):
    queries: str
    task_id: str

# response_schema：外部resume必须遵守这个结构
class TaskReview(BaseModel):
    approved: bool
    revised_query: str | None = None
    note: str = ""

def planner(state: OverallState) -> Command[Literal["worker"]]:
    sends = [
        Send("worker", WorkerState(queries=q, task_id=f"task-{i}"))
        for i, q in enumerate(state["queries"])
    ]
    return Command(goto=sends)

def worker(state: WorkerState):
    review = interrupt(
        value={
            "task_id": state["task_id"],
            "original_query": state["queries"],
            "hint": "审核这条查询，可修改query"
        },
        response_schema=TaskReview
    )

    if review.approved:
        final_q = review.revised_query or state["queries"]
        return {
            "results": [f"ok:{final_q}, note={review.note}"]
        }
    else:
        return {"results": [f"reject:{state['queries']}"]}

builder = StateGraph(OverallState)
builder.add_node("planner", planner)
builder.add_node("worker", worker)
builder.add_edge(START, "planner")
builder.add_edge("worker", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
cfg = {"configurable": {"thread_id": "hitl-demo-1"}}

if __name__ == "__main__":
    # 1. 启动，并行触发多个interrupt
    state = {"queries": ["查询A", "查询B"]}
    print("[start state]", state)
    tmp_state = graph.invoke(state, cfg)

    # 2. 获取 interrupts 并组装成 resume_payload
    resume_payload = {}
    interrupt_list = tmp_state["__interrupt__"]
    for it in interrupt_list:
        # 模拟不同分支的人工介入情况
        if it.value["task_id"] == "task-0":
            # task-0 通过
            resume_payload[it.id] = TaskReview(
                approved = True, 
                revised_query = "修改A",
                note = "ok"
            )
        else:
            # 其他 task 不通过
            resume_payload[it.id] = TaskReview(
                approved = False, 
                note = "拒绝"
            )

    print("[resume_payload]", resume_payload)

    # 3. 并行多个中断
    state = graph.invoke(
        Command(resume=resume_payload),
        config=cfg
    )
    print("[final state]", state)
