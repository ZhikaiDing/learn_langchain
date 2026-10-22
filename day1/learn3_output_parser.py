from langchain_core.output_parsers import (
    JsonOutputParser, PydanticOutputParser, XMLOutputParser,
    MarkdownListOutputParser
)
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel, Field

DOTENV_PATH = "day1/.env"
NEED_TRY_FORMAT = True
NEED_MODEL_RUN = False
INPUT = {
    "user_query": "你好",
    "format_instruct": ""
}
SYS_PROMPT = """回复时严格输出 xml 格式，无需额外文字。{format_instruct}"""

load_dotenv(DOTENV_PATH)

#%% output format
parser = XMLOutputParser(tags=["foo", "bar", "baz"])
INPUT["format_instruct"] = parser.get_format_instructions()

if NEED_TRY_FORMAT:
    class MyPydOut(BaseModel):
        name: str = Field(description="角色名称")
        background: str = Field(description="角色背景简述（不超过300字）")

    parsers = {
        "[json]": JsonOutputParser(),
        "[pydantic]": PydanticOutputParser(pydantic_object=MyPydOut),
        "[xml]": parser,
        "[list](markdown)": MarkdownListOutputParser()
    }

    for key in parsers:
        parser = parsers[key]
        print("-"*40)
        print(key, parser.get_format_instructions())

    print("="*50)

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
