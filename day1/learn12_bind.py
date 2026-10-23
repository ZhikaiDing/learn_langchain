from langchain_core.runnables import RunnableLambda

def func1(x, a, **kw):
    print("[func1]")
    print("   x:", x)   # "input"
    print("   a:", a)   # "A"
    print("   kw:", kw) # {'b': 'B'}
    return x, kw

def func2(x, *args, **kw):
    print("[func2]")
    print("   x:", x) # 元组, 即 func1 的输出
    print("   args:", args) # 空的
    print("   kw:", kw) # 空的
    return x

chain1 = RunnableLambda(func1)
chain2 = RunnableLambda(func2)

chain_org = chain1 | chain2
chain = chain_org.bind(a="A", b="B") # chain1 会接收到这组参数
# 只作用于 整个chain 的入参

chain.invoke("input")
print("="*50)

chain.invoke("input", a="cur A") # 这里的参数优先级更高
