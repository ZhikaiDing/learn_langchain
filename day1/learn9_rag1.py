from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

#%% docs
doc1 = """
LangChain 是大语言模型应用开发框架。
LCEL 是 LangChain 表达式语言，用来组装可运行链路。
RAG 检索增强生成，先检索外部文档再交给大模型回答。
RecursiveCharacterTextSplitter 是递归文本切分器，按标点分层切割文本。
FAISS 是Meta开源的向量相似度检索库，内存向量库。
"""

doc2 = """## 为什么废弃 RunnableWithMessageHistory？

1. **RWH 能力局限**
   - 只能自动存 HumanMessage + AIMessage；**原生不支持 ToolMessage**，做 Agent 工具调用场景就会失效，需要自己手动补消息存储
   - 只能保存消息列表，**不能保存其他状态**（中间检索结果、变量、断点、临时参数）
   - 没有 “状态快照、时间旅行、中断恢复” 这些 Agent 必备能力
2. **LangGraph Checkpointer（检查点）是下一代方案**
LangGraph 的持久化不是单独的 “History 包装器”，而是**整个 graph state 的快照持久化**：
   - 状态里面可以放 `messages`（消息列表），也可以放任意业务字段
   - 每执行完一个 node，自动打 checkpoint 快照；通过`thread_id`（等价旧的 session_id）隔离会话
   - 内置多种后端：内存`MemorySaver`（demo）、SqliteSaver、PostgresSaver，**生产直接对接数据库**
   - Agent 场景（ToolCall、多轮循环、人工介入 interrupt）原生支持，所有消息 Human/AI/Tool 全部自动进 state 并持久化，这是 RWH 做不到的Langchain

> 
> 一句话区分：
> 
> 
> - RWH：**LCEL 链的外挂记忆**，只管理消息，只适合简单单轮对话链
> - LangGraph Checkpointer：**整个应用状态的持久化**，消息只是 state 中的其中一个字段，Agent 优先选它
"""

docs = [
    Document(doc1), Document(doc2)
]

#%% split docs
spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

split_docs = spliter.split_documents(docs)

i = 0
for d in split_docs:
    i += 1
    print(f"[doc {i}]", d, "\n")

#%% 向量库
DOTENV_PATH = "day1/.env"
load_dotenv(dotenv_path=DOTENV_PATH)

embeddings = OpenAIEmbeddings(
    base_url=os.getenv("ALI_URL"),
    model=os.getenv("ALI_EBD_MODEL"),
    api_key=os.getenv("ALI_API_KEY"),
    check_embedding_ctx_length=False
)

vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
retiever = vector_store.as_retriever(search_kwargs={"k":3})

ret = retiever.invoke("什么是LCEL？")
i = 0
print("[vec db ret]")
for d in ret:
    i += 1
    print(f"  [doc {i}]", d, "\n")
