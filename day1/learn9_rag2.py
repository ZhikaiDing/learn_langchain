from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore

#%% prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识库问答助手，只根据下面提供的上下文回答问题。如果上下文没有答案就如实说不知道。\n上下文：{context}"),
    ("human", "用户问题：{question}")
])

#%% model
model = ChatOpenAI(
    api_key=os.getenv("DS_API_KEY"),
    base_url=os.getenv("DS_URL"),
    model=os.getenv("DS_MODEL"),
    temperature=1 # 这个模型必须设置为1
)

#%% embedding
DOTENV_PATH = "day1/.env"
load_dotenv(dotenv_path=DOTENV_PATH)

embeddings = OpenAIEmbeddings(
    base_url=os.getenv("ALI_URL"),
    model=os.getenv("ALI_EBD_MODEL"),
    api_key=os.getenv("ALI_API_KEY"),
    check_embedding_ctx_length=False
)

#%% vector store | retriever
# 连接已经建好的、持久化的向量库
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="knowledge_base_01",
    url="http://127.0.0.1:6333"
)

retriever = vector_store.as_retriever(search_kwargs={"k":4})

#%% reranker
# 增加Reranker（生产强烈建议）
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

compressor = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge‑reranker‑v2‑m3"),
    top_n=2
)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

#%% chain
def format_docs_with_source(docs):
    """带上溯源元数据"""
    lines = []
    for d in docs:
        source = d.metadata.get("source","unknown")
        lines.append(f"【来源：{source}】\n{d.page_content}")
    return "\n\n".join(lines)

rag_chain_prod = (
    {"question": RunnablePassthrough(),
     "context": compression_retriever | format_docs_with_source}
    | rag_prompt
    | model
    | StrOutputParser()
)

resp = rag_chain_prod.invoke("什么是LCEL？")
print(resp)
