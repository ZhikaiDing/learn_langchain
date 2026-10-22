from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

params = {
    "query": "你叫什么？"
}

msgs = [
    ("system", "xxx"),
    ("human", "你好，我是{user_name}, {query}")
]

prompt = ChatPromptTemplate.from_messages(msgs)

#%% 加参数
# 方案1
def add_name_base(d:dict):
    d["user_name"] = "abc" # 改变原始 dict
    return d
add_name1 = RunnableLambda(add_name_base)

chain1 = add_name1 | prompt
ret = chain1.invoke(params)
print("[message1]", ret, "\n"+"="*50)

# 方案2
print("[org dict]", params)
add_name2 = RunnablePassthrough.assign(
    user_name = RunnableLambda(lambda d: "ABC"),
    more = RunnableLambda(lambda d: "test2")
) # 不改变原始 dict

chain2 = add_name2 | prompt

ret = chain2.invoke(params)
print("[cur dict]", params, "\n")
print("[message2]", ret, "\n"+"="*50)

# 方案3
def add_name_base2(d:dict):
    d["user_name"] = "def" # 改变原始 dict
    return "DEF" # 返回值不影响 RunnablePassthrough 
add_name3 = RunnablePassthrough(add_name_base2)
chain3 = add_name3 | prompt

ret = chain3.invoke(params)
print("[cur dict]", params, "\n")
print("[message3]", ret, "\n"+"="*50)

#%% 透传全部参数
all_info = RunnableParallel(
    {
        "org_input": RunnablePassthrough(),
        "message": chain3
    }
)

ret = all_info.invoke(params)
print("[all info]", ret, "\n"+"="*50)

#%% 透传任意参数
chain4 = prompt | RunnablePassthrough() # 这里透传 message 列表
ret = chain4.invoke(params)
print("[message4]", ret, "\n"+"="*50)
