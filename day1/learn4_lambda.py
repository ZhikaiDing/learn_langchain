from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage

input_dict = {
    "query": "你好"
}

#%% prompt
msgs = [
    ("system", "xxx"),
    ("human", "{query}")
]

prompt = ChatPromptTemplate.from_messages(msgs)

#%% runnable labmda
def deal_prompt(prompt_ret:PromptValue) -> list[BaseMessage]:
    return prompt_ret.to_messages()

pre_process = RunnableLambda(deal_prompt)
# 注意: 输入/输出的参数类型 需要符合 管道上/下游的参数类型

#%% chain & run
chain = prompt | pre_process
ret = chain.invoke(input_dict)
print(ret)
