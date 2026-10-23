from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DOTENV_PATH = "day1/.env"
PARAMS = {
    "query": "FastAPI"
}

#%% output format
class Answer(BaseModel):
    concept:str = Field(description="概念名称")
    definition:str = Field(description="简短定义（不超过100字）")
    key_points:list[str] = Field(description="关键点列表（每点不超过50字）")

parser = PydanticOutputParser(pydantic_object=Answer)
answer_format=parser.get_format_instructions()
print(answer_format)

#%% messages
org_msgs = [
    ("system", "严格输出 JSON，不要多余文字。格式要求：{answer_format}"),
    ("human", "解释 {query}")
]
# 注: **`("system", "xxx")` 只是语法糖，仅用于 `ChatPromptTemplate.from_messages()`**，它会在内部被翻译成真正的消息对象

prompt = ChatPromptTemplate.from_messages(org_msgs).partial(
    answer_format=answer_format
)
msgs = prompt.format_messages(**PARAMS)
print("[messages]")
for msg in msgs:
    print(" ", msg)

#%% model
load_dotenv(dotenv_path=DOTENV_PATH)
print("load model:", os.getenv("DS_URL"))
model = ChatOpenAI(
    api_key=os.getenv("DS_API_KEY"),
    base_url=os.getenv("DS_URL"),
    model=os.getenv("DS_MODEL"),
    temperature=1 # KIMI 模型必须设置为1
)

#%% chain
chain = prompt | model | parser
res = chain.invoke(PARAMS) # 直接是python对象，不是字符串！
# 等价于： msgs = prompt.invoke(PARAMS); mdl_ans = model.invoke(msgs); res = parser.invoke(mdl_ans)
print(res)
