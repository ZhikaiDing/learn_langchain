from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json

DOTENV_PATH = "day1/.env"
NEED_MODEL_RUN = True
INPUT = {
    "user_query": "你好",
    "format_instruct": ""
}
SYS_PROMPT = """回复时严格输出 xml 格式，无需额外文字。{format_instruct}"""

load_dotenv(DOTENV_PATH)

#%% output format
parser = XMLOutputParser()
INPUT["format_instruct"] = parser.get_format_instructions()

#%% messages
msgs = [
    ("system", SYS_PROMPT),
    ("human", "{user_query}")
]

prompt = ChatPromptTemplate.from_messages(msgs)
print("[messages]", prompt.format_messages(**INPUT))

#%% model
model = ChatOpenAI(
    base_url=os.getenv("KIMI_URL"),
    model=os.getenv("KIMI_MODEL"),
    api_key=os.getenv("KIMI_API_KEY"),
    temperature=1
)

#%% chain
chain = prompt | model

if NEED_MODEL_RUN:
    #%% run
    res = chain.invoke(INPUT)
    print(f"[content]({type(res.content)})",res.content)

    #%% parse
    parsed = parser.parse(res.content)
    # 或写成 parsed = parser.invoke(res)
    print(f"[parsed]({type(parsed)})", json.dumps(parsed, indent=2, ensure_ascii=False))
