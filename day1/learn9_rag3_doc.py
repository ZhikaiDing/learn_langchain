#%% Document
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain是LLM应用开发框架，支持RAG与Agent",
    metadata = {"source":"intro.txt", "page":1}
)
print("[Document]", doc.page_content)
print("[Document]", doc.metadata)
print("="*50 + "\n")

#%% Loader
from langchain_community.document_loaders import TextLoader,PyPDFLoader

# 1. txt加载器
loader_txt = TextLoader(
    "res/demo_doc.txt", encoding="utf-8"
)
docs_txt = loader_txt.load()  # 返回 List[Document]
print("[txt]", docs_txt[0].page_content)
print("[txt]", docs_txt[0].metadata)
print("-"*50 + "\n")

# 2. PDF加载器（一页一个Document）
loader_pdf = PyPDFLoader("~/Documents/article/time_seq/Con4m: Context-aware Consistency Learning.pdf")
docs_pdf = loader_pdf.load()
# docs_pdf[0].metadata 自带 page、source
print("[pdf]", docs_pdf[0].page_content)
print("[pdf]", docs_pdf[0].metadata)
print("="*50 + "\n")

#%% Spliter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 拿到加载好的文档（List[Document]）
raw_docs = docs_pdf

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每个块最大字符数
    chunk_overlap=50,      # 块之间重叠50字符，保留上下文
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

# 执行切分，返回 List[Document]
split_docs = splitter.split_documents(raw_docs)

# 查看第一个分块
print("[Spliter]", split_docs[0].page_content)
print("[Spliter]", split_docs[0].metadata)
