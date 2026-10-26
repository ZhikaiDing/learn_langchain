from pydantic import BaseModel, Field
from typing import Annotated, Any
import random

from langgraph.graph.message import add_messages
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END

NEED_RUN = False
NEED_SHOW_GRAPH_IMG = True
NEED_SAVE_GRAPH_IMG = False
SAVE_GRAPH_IMG = "tmp/graph.png"

simple_reducer = lambda old, new: old + new

class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    cur_input: str = ""
    cur_output: str = "" # 输出类型1: 模型回复文本
    cur_toolcall: list = [] # 输出类型2: 模型回复 tool_calls
    cur_tool_rslt: str = "" # 输出类型3: 工具调用结果

    max_loop: int = 3 # 循环上限
    loop_count: Annotated[int, simple_reducer] = 0

    # 次数统计
    toolcall_count: Annotated[int, simple_reducer] = 0
    llm_count: Annotated[int, simple_reducer] = 0

def input_addto_msgs(state:State):
    return {"messages": [HumanMessage(state.cur_input)]}

def fake_llm(state:State):
    if random.random() < 0.4:
        # 模拟: 模型回复文本
        llm_ans = "模型输出..."
        return {
            "cur_output": llm_ans,
            "cur_toolcall": [],
            "messages": [AIMessage(llm_ans)],
            "loop_count": 1,
            "llm_count": 1
        }
    else:
        # 模拟: 模型回复 tool_calls
        llm_tc_ans = [{
            "name": "show_toolcall", 
            "args":{}, 
            "id": "t1", 
            "type": "tool_call"
        }]
        return {
            "cur_output": "",
            "cur_toolcall": llm_tc_ans,
            "messages": [AIMessage(
                content="",
                tool_calls=llm_tc_ans
            )],
            "loop_count": 1,
            "llm_count": 1
        }

def toolcall(state:State):
    tool_name = state.cur_toolcall[0]["name"]
    tool_id = state.cur_toolcall[0]["id"]
    print(f"[debug] call {tool_name}()")
    tool_ret = "use toolcall"
    return {
        "cur_tool_rslt": tool_ret,
        "messages": [ToolMessage(tool_ret, tool_call_id=tool_id)],
        "toolcall_count": 1
    }

graph_builder = StateGraph(State)

graph_builder.add_node("user_input_to_msgs", input_addto_msgs)
graph_builder.add_node("llm", fake_llm)
graph_builder.add_node("toolcall", toolcall)

graph_builder.add_edge(START, "user_input_to_msgs")
graph_builder.add_edge("user_input_to_msgs", "llm")

# 条件边 - 写法1: path (返回分支名) + path_map (分支名:结点名)
graph_builder.add_conditional_edges(
    "llm",
    lambda state: "finish" if state.cur_output else "tool",
    {
        "tool": "toolcall",
        "finish": END
    }
)
graph_builder.add_conditional_edges(
    "toolcall", 
    lambda state: "llm" if state.loop_count < state.max_loop \
        else "finish",
    {
        "llm": "llm",
        "finish": END
    }
)

# # 条件边 - 写法2: path (直接返回结点名)
# # tips: 图的展示有遗漏
# graph_builder.add_conditional_edges(
#     "toolcall", 
#     lambda state: "llm" if state.loop_count < state.max_loop\
#         else END
# )

graph = graph_builder.compile()

#%% 展示图
# 简单展示
graph.get_graph().print_ascii() # 无论如何, 展示的图结构不完整

# 在线展示
graph_str = graph.get_graph().draw_mermaid()
print("`"*50)
print(graph_str) # 在线展示 https://mermaid.live/ (登录需翻墙)
print("`"*50)

# 本地展示图片
if NEED_SHOW_GRAPH_IMG or NEED_SAVE_GRAPH_IMG:
    from PIL import Image
    import io

    png_bytes = graph.get_graph().draw_mermaid_png()
    img = Image.open(io.BytesIO(png_bytes))

    if NEED_SHOW_GRAPH_IMG:
        img.show() # 唤起系统图片查看器
    if NEED_SAVE_GRAPH_IMG:
        img.save(SAVE_GRAPH_IMG)

# %% 运行图
if NEED_RUN:
    import json
    from copy import deepcopy

    try_times = 5
    for i in range(try_times):
        state = {
            "cur_input": "",
            "max_loop": 3
        }

        state["cur_input"] = f"测试轮次 {i}"
        state = graph.invoke(state)

        st = deepcopy(state)
        del st["messages"]
        print(json.dumps(st, indent=4, ensure_ascii=False))

        print("-"*40)
        print("[final messages]")
        for msg in state["messages"]:
            print(
                f"  ({type(msg).__name__})", 
                msg.content if msg.content else msg.tool_calls
            )
            print()
        print("="*50)
