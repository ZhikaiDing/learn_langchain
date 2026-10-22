from langchain_core.runnables import (
    RunnableBranch, RunnableLambda, RunnablePassthrough
)
import random

INPUT = {
    "query": "你好",
    "run_times": 10
}

TAGS = ["A", "B", "C"]

get_tag = RunnableLambda(lambda d: random.choice(TAGS))
chainA = RunnableLambda(lambda tag: "call A")
chainB = RunnableLambda(lambda tag: "call B")
chainC = RunnableLambda(lambda tag: "call C")

branch = RunnableBranch(
    (lambda tag: tag=="A", chainA), # if
    (lambda tag: tag=="B", chainB), # else if
    chainC # else
)
# 命中第一个 true 就停止

chain = RunnablePassthrough.assign(ret = get_tag | branch)

for i in range(INPUT["run_times"]):
    ret = chain.invoke(INPUT)
    print(f"[{i}]", ret)
