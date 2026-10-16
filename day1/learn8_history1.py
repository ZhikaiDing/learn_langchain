from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage

org_msg = [
    ("system", "你是智能助手。回答用户问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]

prompt = ChatPromptTemplate.from_messages(org_msg)
fake_llm = RunnableLambda(lambda msgs: AIMessage(content="思考中..."))

chain = prompt | RunnablePassthrough(lambda msgs: print("[messages]", msgs, "\n")) | fake_llm

store = {}
def get_session_hist(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

test_hist = RunnableWithMessageHistory(
    chain,
    get_session_hist,
    input_messages_key="input",
    history_messages_key="history"
)

inputs = ["你好", "还在吗？"]
for query in inputs:
    ret = test_hist.invoke(
        {"input": query},
        config={"configurable": {"session_id":"sess_001"}}
    )
    print("[ret content]", ret.content, "\n")
    print("[sess_001]", store["sess_001"].messages, "\n"+"="*50)
