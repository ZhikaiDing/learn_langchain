from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda, RunnablePassthrough, RunnableParallel
)

INPUT = {
    "query": "你好"
}

tag = 0
tags = {}

def callback_parents(parent_run_id):
    if not parent_run_id:
        return 0
    
    ret = 0
    cur_pid = parent_run_id
    while cur_pid in tags:
        ret += 1
        cur_pid = tags[cur_pid]["parent"]
    return ret

def callback_print(mode:str, run_id, parent_run_id, info):
    global tag, tags
    if run_id not in tags:
        tags[run_id] = {"tag":tag, "parent":parent_run_id}
        tag += 1

    parent_num = callback_parents(parent_run_id)
    head_str = "  " * parent_num
    tag_str = f"({tags[run_id]["tag"]})"
    space_str = " " * len(tag_str)
    print(head_str + tag_str + f"[{mode}]")
    print(head_str + space_str + str(info))

class MySimpleCallBack(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        callback_print("chain start", run_id, parent_run_id, inputs)
        return # super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)

    def on_chain_end(self, outputs, *, run_id, parent_run_id = None, **kwargs):
        callback_print("chain end", run_id, parent_run_id, outputs)
        return # super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

callback1 = MySimpleCallBack()

org_msgs = [
    ("system", "你是智能助手，根据用户问题进行回答。"),
    ("human", "{query}")
]
prompt = ChatPromptTemplate.from_messages(org_msgs)

fake_llm = RunnableLambda(lambda x: "思考中...")

chain = RunnableParallel(
    {
        "input": RunnablePassthrough(),
        "output": prompt | fake_llm
    }
)

ret = chain.invoke(
    INPUT, 
    config={"callbacks": [callback1]}
)
# 0: chain
#   1: input
#   2: output
#     3: prompt
#     4: fake_llm

# print("[finish]", ret)