from pydantic import BaseModel
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
import operator
from pydantic import BaseModel

class State(BaseModel):
    hist: Annotated[list, operator.add] = []
    input: str = ""
    human_decision: bool = True
    count: Annotated[int, operator.add] = 0

class ResumeSchema(BaseModel):
    decision_org: str = "Y"
    decision: bool = True

def human_judge(state: State):
    # print("[debug] cur state:", state)
    resume = interrupt(
        value="是否继续？",
        response_schema=ResumeSchema
    )
    print("[debug] human decision:", resume)
    return {
        "human_decision": resume.decision,
        "count": 1,
    }

def finish(state: State):
    return {"hist": [(state.input, state.count)]}

def route(state: State):
    if state.human_decision:
        return "human_judge"
    else:
        return "finish"

graph_builder = StateGraph(State)
graph_builder.add_node("human_judge", human_judge)
graph_builder.add_node("finish", finish)
graph_builder.add_edge(START, "human_judge")
graph_builder.add_conditional_edges(
    "human_judge",
    route,
    {
        "human_judge": "human_judge",
        "finish": "finish",
    },
)
graph_builder.add_edge("finish", END)

ckpt = MemorySaver()
graph = graph_builder.compile(checkpointer=ckpt)
print(graph.get_graph().draw_mermaid())

cfg = {"configurable": {"thread_id": "tid-001"}}

state = {"input": ""}
usr_input = input("输入 Q 结束对话\n你的输入：")

while usr_input != "Q":
    # state["input"] = usr_input # 错误
    state = {"input": usr_input} # 正确
    # tips: 使用 checkpointer 时只传增量, 而不是在上一轮的输出中修改

    ret = graph.invoke(state, config=cfg)

    while "__interrupt__" in ret:
        print("[debug] ret:", ret)
        hum_judge = input("是否继续? (Y/N) ")
        resume = ResumeSchema(
            decision_org = hum_judge,
            decision = hum_judge.strip().upper() == "Y"
        ) # 这里的 resume 也可以是 dict
        ret = graph.invoke(Command(resume=resume), config=cfg)

    # 只保留业务状态，避免把 __interrupt__ 带进下一轮
    state = {k: v for k, v in ret.items() if k != "__interrupt__"}

    print(state)
    print("-" * 40)
    usr_input = input("输入 Q 结束对话\n你的输入：")

print("=" * 50)
print(state)
