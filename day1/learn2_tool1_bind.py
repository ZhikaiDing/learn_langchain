from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

DOTENV_PATH = "day1/.env"
NEED_CALL_LLM = True

#%% tool
@tool
def add(a:float, b:float) -> float:
    """两数相加
    Args:
        a: 第一个数
        b: 第二个数
    """
    return a + b

tools = [add, ]

def show_tool(tool_func):
    print("[tool name]", tool_func.name)
    print("[args]", json.dumps(tool_func.args, indent=2, ensure_ascii=False))
    print("[description]", tool_func.description)

def show_tools(tools, mode=0):
    if mode == 1:
        for t in tools:
            show_tool(t)
            print()
    elif mode == 0:
        tool_schemas = {
            "tools": [
                convert_to_openai_tool(t) for t in tools
            ]
        }
        print("LLM API 请求体 带如下内容:", json.dumps(tool_schemas, indent=2, ensure_ascii=False))

show_tools(tools)

#%% model
load_dotenv(dotenv_path=DOTENV_PATH)
print("load model:", os.getenv("KIMI_URL"))
model = ChatOpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url=os.getenv("KIMI_URL"),
    model=os.getenv("KIMI_MODEL"),
    temperature=1 # KIMI 模型必须设置为1
)

model = model.bind_tools(tools)
# 在调用LLM API 侧看来
#   .bind_tools 方案不改变 system 提示词，
#   在**API request body**追加 `tools: [schema1, schema2...]`，还可以带上 `tool_choice`

if NEED_CALL_LLM:
    res = model.invoke("11.4加256.5等于？")
    print("[content]", res.content) 
    print("[tool_calls]", json.dumps(res.tool_calls, indent=2, ensure_ascii=False))
# res.content 此时经常是空字符串，
# 模型输出的是 tool_call 结构，而不是自然文本。
