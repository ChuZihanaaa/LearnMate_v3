import os
import time

import fitz
import gradio
import asyncio
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

from functools import lru_cache
import diskcache as dc
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ------------------- 路径配置 -------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(base_dir, "input")
OUTPUT_DIR = os.path.join(base_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
persist_directory = os.path.join(OUTPUT_DIR, "chroma_db")
CACHE_DIR = os.path.join(OUTPUT_DIR, "rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------- 缓存初始化 -------------------
rag_cache = dc.Cache(CACHE_DIR)


# ------------------- PDF 预处理函数 -------------------
def preprocess_pdf(pdf_path: str):
    """处理 PDF 文件"""
    print(f"正在处理 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # 清洗逻辑
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', cleaned_text)

    # 调用通用分块
    return split_and_save_chunks(cleaned_text, os.path.basename(pdf_path))


# --- SRT 字幕处理函数 ---
def preprocess_srt(srt_path: str):
    """
    读取 .srt 字幕文件 -> 去除时间轴和序号 -> 清洗 -> 调用通用分块
    """
    print(f"正在处理字幕: {srt_path}")

    if not os.path.exists(srt_path):
        print(f"错误：文件不存在 {srt_path}")
        return

    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    text_content = []
    for line in lines:
        line = line.strip()
        # 1. 跳过纯数字（字幕序号）
        if line.isdigit():
            continue
        # 2. 跳过时间轴 (格式如 00:00:20,000 --> 00:00:24,400)
        if '-->' in line:
            continue
        # 3. 跳过空行
        if not line:
            continue

        # 将有效文本加入列表
        text_content.append(line)

    # 合并为完整文本（用空格连接，避免单词粘连）
    full_text = " ".join(text_content)

    # 简单的清洗（与 PDF 类似）
    # 去除多余空格
    cleaned_text = re.sub(r'\s+', ' ', full_text)

    # 保存一份清洗后的全文本备份（可选，方便调试）
    base_name = os.path.basename(srt_path)
    with open(os.path.join(OUTPUT_DIR, f"cleaned_{base_name}.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # 调用通用分块函数
    return split_and_save_chunks(cleaned_text, base_name)

# ---通用分块与保存函数 ---
def split_and_save_chunks(text: str, source_filename: str):
    """
    通用函数：接收清洗后的文本，分块并保存。
    文件名格式：chunk_{source_filename}_{i}.txt (避免覆盖)
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # 获取不带后缀的文件名，作为前缀 (例如 "Chapter 2")
    base_name = os.path.splitext(os.path.basename(source_filename))[0]
    # 清理文件名中的空格，避免后续处理麻烦
    base_name = base_name.replace(" ", "_")

    print(f"正在为 {base_name} 生成分块...")
    count = 0
    for i, chunk in enumerate(chunks):
        # 关键修改：文件名包含来源标识
        out_name = f"chunk_{base_name}_{i}.txt"
        with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(chunk)
        count += 1

    print(f"[{source_filename}] 处理完成，生成 {count} 个分块。")
    return chunks

# ------------------- 构建向量库 -------------------
def build_vectorstore():
    chunks = []
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("chunk_") and f.endswith(".txt"):
            with open(os.path.join(OUTPUT_DIR, f), "r", encoding="utf-8") as file:
                chunks.append(file.read())

    if not chunks:
        raise FileNotFoundError("未找到分块文件")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=persist_directory)
    print("Chroma 数据库构建完成")
    return vectorstore


# ------------------- 初始化 RAG 组件 -------------------
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=500
)

prompt_template = """
你是课程学习助手。

按以下步骤推理：
1. 识别上下文中的相关信息。
2. 总结或推测答案。
3. 注明来源块编号（如 chunk_3）。

示例：
- 问题: 什么是敏捷开发？
- 上下文: chunk_10: Agile means iterative development...
- 答案: 敏捷开发是一种迭代和增量开发的软件开发方法。
- 来源: chunk_10

基于以下课程上下文，回答问题。优先提取关键信息，若信息不足，可基于相关概念推测或总结，并注明来源块编号。若完全无法回答，说“未知”。

上下文: {context}
问题: {question}
答案:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 30}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


# ------------------- 缓存 RAG 函数 -------------------
def cached_rag(query: str) -> str:
    normalized = query.strip().lower()
    cache_key = f"rag_v1:{hashlib.md5(normalized.encode()).hexdigest()}"

    if cache_key in rag_cache:
        print(f"[缓存命中] {cache_key[-8:]}")
        return rag_cache[cache_key]

    print(f"[缓存未命中] 执行 RAG...")
    result = qa_chain.invoke({"query": query})["result"]
    rag_cache[cache_key] = result
    return result


# ------------------- 健康检查 -------------------
def health_check():
    return {
        "status": "healthy",
        "cache_size": len(rag_cache),
        "vector_count": vectorstore._collection.count()
    }