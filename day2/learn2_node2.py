from pydantic import BaseModel
from typing import TypedDict, Annotated
import operator

from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    info: Annotated[list, operator.add] = []
    count: Annotated[int, operator.add] = 0

@dataclass
class MyContext:
    user_id: str
    tenant_id: str

def node1(state:State, config:RunnableConfig):
    i = state['count']
    tid = config['configurable']['thread_id']
    return {
        "info": [f"{i} | node1 | config thread id: {tid}"],
        "count": 1
    }

def node2(state:State, runtime: Runtime[MyContext]):
    i = state['count']
    uid = runtime.context.user_id
    return {
        "info": [f"{i} | node2 | runtime user id: {uid}"],
        "count": 1
    }

graph_builder = StateGraph(State, context_schema=MyContext)
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", END)

ckpt = MemorySaver()
graph = graph_builder.compile(checkpointer=ckpt)

import json
cfg = {"configurable": {"thread_id": "th-001"}}
ctx = MyContext(user_id="u-001", tenant_id="te-001")
for i in range(3):
    print("-"*40)
    state = graph.invoke({}, config=cfg, context=ctx)
    print(json.dumps(state, indent=2, ensure_ascii=False))
