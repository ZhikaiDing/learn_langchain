from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
import os

DOTENV_PATH = "day1/.env"
load_dotenv(DOTENV_PATH)   # 必须在 import 链和 invoke 之前执行
print(os.getenv("LANGSMITH_TRACING"))



chain1 = RunnableLambda(lambda _: "[prompt]")\
.with_config(
    tags = ["prompt"]
)

chain2 = RunnableLambda(lambda _: "[model]")\
.with_config(
    tags = ["model"],
    metadata = {"model": "gpt-4o"}
)

chain = (
    chain1 | RunnablePassthrough(lambda x: print(x)) 
    | chain2 | RunnablePassthrough(lambda x: print(x))
)

ret = chain.invoke("input")
