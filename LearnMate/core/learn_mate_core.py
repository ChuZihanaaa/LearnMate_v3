import os
import time

import fitz
import shutil
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

import diskcache as dc
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

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

# ------------------- 初始化 RAG 组件 -------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

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


def _refresh_retriever():
    """重建检索器与 QA Chain，确保新增文档后立即可用。"""
    global retriever, qa_chain
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


_refresh_retriever()


# ------------------- 构建/重建向量库 -------------------
def build_vectorstore_from_output() -> int:
    """
    从 OUTPUT 目录的 chunk_*.txt 重新构建向量库。
    返回：载入的分块数量。
    """
    texts = []
    metadatas = []

    for fn in sorted(os.listdir(OUTPUT_DIR)):
        if fn.startswith("chunk_") and fn.endswith(".txt"):
            path = os.path.join(OUTPUT_DIR, fn)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            chunk_id = fn.replace(".txt", "")
            texts.append(f"来源ID: {chunk_id}\n{content}")
            metadatas.append({"source": fn})

    if not texts:
        raise FileNotFoundError("未找到分块文件，请先上传或初始化数据。")

    # 重建持久化向量库
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    global vectorstore
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding_function=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    _refresh_retriever()
    print(f"Chroma 数据库重建完成，载入 {len(texts)} 个分块")
    return len(texts)


def add_documents_to_vectorstore(chunks: List[str], source_filename: str) -> int:
    """
    将新文档分块添加到向量库，并写入 OUTPUT 目录以持久化。
    返回：新增分块数量。
    """
    global vectorstore
    texts = []
    metadatas = []
    base_name = os.path.splitext(os.path.basename(source_filename))[0].replace(" ", "_")

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{base_name}_{i}"
        text_with_id = f"来源ID: {chunk_id}\n{chunk}"
        texts.append(text_with_id)
        metadatas.append({"source": f"{chunk_id}.txt"})

        out_name = f"{chunk_id}.txt"
        with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(chunk)

    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        _refresh_retriever()
        print(f"✅ 成功添加 {len(texts)} 个分块到向量库")

    return len(texts)


def ingest_file(file_path: str) -> Dict[str, int]:
    """
    处理并写入单个文件到向量库。
    支持 PDF / SRT。
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".pdf":
        chunks = preprocess_pdf(file_path)
    elif file_ext == ".srt":
        chunks = preprocess_srt(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")

    if not chunks:
        raise ValueError(f"未能从文件中提取有效内容: {os.path.basename(file_path)}")

    count = add_documents_to_vectorstore(chunks, os.path.basename(file_path))
    return {"file": os.path.basename(file_path), "chunks": count}


def ingest_files(file_paths: List[str]) -> List[Dict[str, int]]:
    """批量处理文件，返回每个文件的分块统计。"""
    results = []
    for path in file_paths:
        results.append(ingest_file(path))
    return results


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
        "vector_count": vectorstore._collection.count() if hasattr(vectorstore, "_collection") else 0
    }


# ------------------- 个性化练习生成 -------------------
response_schemas = [
    ResponseSchema(name="question", description="题目内容，必须清晰完整"),
    ResponseSchema(name="options",
                   description="包含4个字符串的列表，分别代表A、B、C、D四个选项的描述，不要带前缀"),
    ResponseSchema(name="answer", description="正确选项的字母，仅输出 'A', 'B', 'C', or 'D'"),
    ResponseSchema(name="explanation", description="答案解析，解释正确原因及干扰项错误原因")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


def generate_quiz(topic: str) -> Dict[str, object]:
    """
    基于知识点生成单选题，返回结构化结果。
    """
    if not topic or not topic.strip():
        raise ValueError("请输入具体的知识点")

    docs = retriever.invoke(topic)
    if docs:
        context_text = "\n".join([d.page_content for d in docs[:3]])
    else:
        context_text = "（未检索到具体课程内容，请基于该知识点的通用概念出题）"

    quiz_template = """
    你是一名专业的大学课程出题老师。请针对目标【知识点】出一道单项选择题。

    【参考课程内容】：
    {context}

    【目标知识点】：{topic}

    【出题要求】：
    1. 优先依据【参考课程内容】出题。如果内容中未包含具体细节，请基于该【目标知识点】的专业知识进行补全，确保题目逻辑通顺。
    2. 题目难度适中，适合大学生复习。
    3. 选项（options）必须是包含4个具体描述的列表，不要包含 "A." 等前缀。
    4. 必须严格遵守下方的 JSON 格式输出。

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=quiz_template,
        input_variables=["context", "topic"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "topic": topic})
    data = output_parser.parse(response.content)

    options = data.get("options", [])
    while len(options) < 4:
        options.append("（生成选项不足）")

    return {
        "question": data.get("question"),
        "options": options[:4],
        "answer": data.get("answer"),
        "explanation": data.get("explanation")
    }


# ------------------- 缓存查看/清理 -------------------
def list_cache(limit: int = 50) -> List[Dict[str, str]]:
    """
    返回缓存条目预览。
    仅包含 key 和内容预览，diskcache 本身不记录时间戳，这里按遍历顺序截取。
    """
    items = []
    for idx, key in enumerate(rag_cache.iterkeys()):
        if idx >= limit:
            break
        value = rag_cache.get(key, "")
        preview = value[:120].replace("\n", " ")
        items.append({"key": key, "preview": preview, "length": len(value)})
    return items


def clear_cache(key: Optional[str] = None) -> int:
    """
    清理缓存；若传入 key 则删除指定条目，否则清空全部。
    返回剩余缓存条目数量。
    """
    if key:
        rag_cache.pop(key, None)
    else:
        rag_cache.clear()
    return len(rag_cache)