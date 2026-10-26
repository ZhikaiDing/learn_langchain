from pydantic import BaseModel, Field
from typing import Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

#%% 关于 State
# 自定义 Reducer（归约器）
def count_up(old_count, incr_count):
    return old_count + incr_count

# 定义 State 数据格式
class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(
        default_factory=list
    )
    cur_input: str = ""
    cur_output: str | None = None
    cur_toolcall: list | None = None
    llm_use_count: Annotated[int, count_up] = 0

# 底层 langchain
fake_llm = RunnableLambda(lambda _: AIMessage("思考中..."))

# 结点函数1: 用户输入 加到 messages 中
def user_query_to_message(state:State):
    return {"messages": [HumanMessage(state.cur_input)]}

# 结点函数2: 调用 runnable 对象
def fake_llm_node(state:State):
    model_ans = fake_llm.invoke(state.messages)
    return {
        "messages": [model_ans],
        "cur_output": model_ans.content,
        "cur_toolcall": model_ans.tool_calls,
        "llm_use_count": 1
    }

#%% 测试代码
# 1. 创建图构建器
builder = StateGraph(State)

# 2. 添加节点
builder.add_node("user_query_to_message", user_query_to_message)
builder.add_node("fake_llm_node", fake_llm_node)

# 3. 连接边：START -> 用户消息转换 -> 调用模型 -> END
builder.add_edge(START, "user_query_to_message")
builder.add_edge("user_query_to_message", "fake_llm_node")
builder.add_edge("fake_llm_node", END)

# 4. 编译成可执行图
graph = builder.compile()

# 5. 运行
inputs = ["测试1", "测试2"]

state = {"cur_input": ""}
for input in inputs:
    state["cur_input"] = input
    state = graph.invoke(state) # 覆盖老的 state

    print("-"*30)
    print("[cur_input]", state["cur_input"])
    print("[cur_output]", state["cur_output"])
    print("[cur_toolcall]", state["cur_toolcall"])
    print("[llm_use_count]", state["llm_use_count"])
    # print("[debug messages]", state["messages"])

print("="*50)
print("[messages]", state["messages"], "\n")
