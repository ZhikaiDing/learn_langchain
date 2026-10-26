from pydantic import BaseModel, Field
from typing import Annotated, Literal

from langchain_core.messages import (
    HumanMessage, AIMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

import asyncio

class State(BaseModel):
    sys_prompt:str = ""
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    cur_input: str = ""
    cur_search: str = ""
    cur_prompt: list = []

    cur_output: str = ""
    cur_toolcall: list | None = None

    llm_count: Annotated[int, lambda old, new: old+new] = 0

# tips: Runnable 对象可作为 node
input_to_msgs = RunnableLambda(lambda state: {
    "messages": [HumanMessage(state.cur_input)]
})

async def fake_search(state:State):
    # use: state.cur_input
    await asyncio.sleep(1.0)
    search_ret = f"[tag {state.llm_count}] LCEL 是 LangChain 表达式语言，用来组装可运行链路。"
    return {"cur_search": search_ret} # tips: 返回增量 dict

def msgs_to_prompt(state: State) -> Command[Literal["call_llm"]]:
    # tips: 通过 Command[Literal["xxx"]] 显示指定下个结点 
    #   可以保证正确展示图结构
    #   如果不写，展示的图结构是有缺失的
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", state.sys_prompt),
            MessagesPlaceholder("messages"),
        ]
    )
    cur_prompt = prompt.invoke(
        {
            "cur_search": state.cur_search,
            "messages": state.messages,
        }
    )

    return Command(
        update = {"cur_prompt": cur_prompt},
        goto = "call_llm"
    ) # tips: 返回增量 Command, 指定下个节点

async def fake_llm(state:State) -> Command[Literal["__end__"]]:
    # tips: goto 只有 END 时, 可省略 -> Command
    # tips: 显示写法需要写成 "__end__" 而非 END
    # use: state.cur_prompt
    await asyncio.sleep(1.0)
    llm_ans = "思考中..."
    return Command(
        update = {
            "cur_output": llm_ans,
            "messages": [AIMessage(llm_ans)],
            "llm_count": 1
        },
        goto = END # tips: END 是图运行的终点
    )

#%% 测试代码
graph_builder = StateGraph(State)

graph_builder.add_node("user_query_to_msgs", input_to_msgs)
graph_builder.add_node("rag_search", fake_search)
graph_builder.add_node("msgs_to_prompt", msgs_to_prompt)
graph_builder.add_node("call_llm", fake_llm)

graph_builder.add_edge(START, "user_query_to_msgs")
graph_builder.add_edge(START, "rag_search")
graph_builder.add_edge(
    ["user_query_to_msgs", "rag_search"], 
    "msgs_to_prompt"
)# tips: 等待两个节点的结果

# graph_builder.add_edge("msgs_to_prompt", "call_llm")
# graph_builder.add_edge("call_llm", END)
# tips: 实际上 Command 指定过下个节点的, 无需 add_edge

graph = graph_builder.compile()
graph.get_graph().print_ascii() # 展示图结构

async def run_graph(state):
    return await graph.ainvoke(state)

# 运行
sys_prompt_str = """你是智能助手，根据资料和对话历史回答用户当前问题。
# 资料
{cur_search}
"""

inputs = ["用户问题1", "用户问题2"]

state = {
    "sys_prompt": sys_prompt_str,
    "cur_input": ""
}
for input in inputs:
    state["cur_input"] = input
    state = asyncio.run(run_graph(state)) 

    print("-"*30)
    # print("[debug state]", state)
    print("[sys_prompt]", state["sys_prompt"])
    print("[cur_input]", state["cur_input"])
    print("[cur_search]", state["cur_search"])
    print("[cur_output]", state["cur_output"])
    print("[llm_count]", state["llm_count"])
    print("[debug cur_prompt]", state["cur_prompt"])
    # print("[debug messages]", state["messages"])

print("="*50)
print("[messages]", state["messages"], "\n")
