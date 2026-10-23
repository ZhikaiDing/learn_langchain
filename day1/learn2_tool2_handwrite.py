from langchain_core.tools import tool, render_text_description
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
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
tool_map = {t.name: t for t in tools}  # 工具名字 -> 工具实例映射

#%% prompt
tool_desc = render_text_description(tools)
system_prompt = f"""你可以调用下面工具：
{tool_desc}

需要调用工具时，**只输出纯JSON，不要任何多余解释**，格式：
{{"name":"工具名","arguments":{{"a":1,"b":2}}}}
不需要调用工具就直接回答用户问题。
"""
print("[sys prompt]", system_prompt)

messages = [
    SystemMessage(content=system_prompt),
    # HumanMessage(content="11.4加256.5等于？")
    HumanMessage(content="你好")
]

#%% model
load_dotenv(dotenv_path=DOTENV_PATH)
print("load model:", os.getenv("KIMI_URL"))
model = ChatOpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url=os.getenv("KIMI_URL"),
    model=os.getenv("KIMI_MODEL"),
    temperature=1 # KIMI 模型必须设置为1
)

if NEED_CALL_LLM:
    res = model.invoke(messages)
    print("[content]", res.content) 

    #%% output parser
    parser = JsonOutputParser()
    tool_call_info = parser.parse(res.content)
    print("[parse content]", json.dumps(tool_call_info, indent=2, ensure_ascii=False))
