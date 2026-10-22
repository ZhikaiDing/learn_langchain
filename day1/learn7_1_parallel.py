from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

params = {
    "user": "ABC",
    "query": "你好"
}

# 显式写法
chain1 = RunnableParallel(
    {
        "org_input": RunnablePassthrough(),
        "norm_name": RunnableLambda(lambda d: d["user"].lower())
    }
)
print(chain1.invoke(params), "\n"+"="*50)

# 隐式写法 (需要左侧或右侧有 runnable obj)
chain2 = {
    "org_input": RunnablePassthrough(),
    "norm_name": lambda d: d["user"].lower(),
    "tag": lambda d: "chain2"
} | RunnablePassthrough()
print(chain2.invoke(params), "\n"+"="*50)
