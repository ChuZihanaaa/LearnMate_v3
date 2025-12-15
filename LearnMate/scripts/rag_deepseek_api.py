import os
import time
import gradio
import asyncio
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from functools import lru_cache
import diskcache as dc
import hashlib


from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

# ------------------- 1. 环境 & 路径 -------------------
load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
print(f"base_dir: {base_dir}")

OUTPUT_DIR = os.path.join(base_dir, os.getenv("OUTPUT_DIR", "output/"))
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

persist_directory = os.path.join(OUTPUT_DIR, "chroma_db")
api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------- 2. 创建/加载 Chroma 向量库 -------------------
if not os.path.exists(persist_directory):
    print("正在构建新的 Chroma 数据库...")
    chunks = []
    texts = []
    metadatas = []

    if os.path.isdir(OUTPUT_DIR):
        # 遍历 output 目录下的所有 txt 分块
        for fn in sorted(os.listdir(OUTPUT_DIR)):
            if fn.startswith("chunk_") and fn.endswith(".txt"):
                path = os.path.join(OUTPUT_DIR, fn)

                # 获取不带后缀的 ID，例如 chunk_Chapter_2_0
                chunk_id = fn.replace(".txt", "")

                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()


                    text_with_id = f"来源ID: {chunk_id}\n{content}"

                    texts.append(text_with_id)
                    metadatas.append({"source": fn})

        if not texts:
            raise FileNotFoundError(f"{OUTPUT_DIR} 中未找到 chunk_*.txt，请先确保运行了 init.py")
    else:
        raise FileNotFoundError(f"{OUTPUT_DIR} 不存在")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    print(f"Chroma 数据库构建完成，共载入 {len(texts)} 个分块")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Chroma 数据库已加载")


# 持久化磁盘缓存
CACHE_DIR = os.path.join(OUTPUT_DIR, "rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
rag_cache = dc.Cache(CACHE_DIR, disk_min_file_size=0)  # 持久化磁盘缓存
print(f"RAG 缓存目录: {CACHE_DIR}")

# ------------------- 3. DeepSeek LLM -------------------
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY 未设置")

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=500
)
print("DeepSeek LLM 配置成功")

# ------------------- 4. Prompt -------------------
prompt_template = """
你是课程学习助手。

按以下步骤推理：
1. 识别上下文中的相关信息。
2. 总结或推测答案。
3. 注明来源块编号（如 chunk_3）。

示例：
- 问题: 什么是敏捷开发？
- 上下文: chunk_Chapter_2_10: Agile means iterative development...
- 答案: 敏捷开发是一种迭代和增量开发的软件开发方法。
- 来源: chunk_Chapter_2_10

基于以下课程上下文，回答问题。优先提取关键信息，若信息不足，可基于相关概念推测或总结，并注明来源块编号。若完全无法回答，说“未知”。

上下文: {context}
问题: {question}
答案:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ------------------- 5. MMR 检索器（动态 k） -------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",                 # Maximum Marginal Relevance
    search_kwargs={"k": 15, "fetch_k": 30}   # k=15 最终返回，fetch_k=30 候选池
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


def cached_rag(query: str) -> str:
    """
    持久化磁盘缓存 RAG 问答（跨进程生效）
    - 第一次运行：完整 RAG → 存入磁盘
    - 第二次运行：直接从磁盘读取 → 0.01s
    """
    # 标准化查询 + 生成唯一 key
    normalized = query.strip().lower()
    cache_key = f"rag_v1:{hashlib.md5(normalized.encode()).hexdigest()}"

    if cache_key in rag_cache:
        print(f"[缓存命中] {cache_key[-8:]}")
        return rag_cache[cache_key]

    print(f"[缓存未命中] 执行 RAG...")
    result = qa_chain.invoke({"query": query})["result"]
    rag_cache[cache_key] = result
    return result


# ==============================================================================
# ------------------- 新增模块：个性化练习生成 (Quiz Generation) -------------------
# ==============================================================================

# 1. 优化 Schema 定义：明确告诉 LLM 选项是一个纯文本列表
response_schemas = [
    ResponseSchema(name="question", description="题目内容，必须清晰完整"),
    ResponseSchema(name="options",
                   description="包含4个字符串的列表（List[str]），分别代表A、B、C、D四个选项的具体描述。注意：不要在字符串里包含 'A.' 或 '1.' 等前缀，只写内容。"),
    ResponseSchema(name="answer", description="正确选项的字母，仅输出 'A', 'B', 'C', or 'D'"),
    ResponseSchema(name="explanation", description="答案解析，解释正确原因及干扰项错误原因")
]

# 初始化解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


def generate_quiz_func(topic: str):
    """
    功能：基于输入的知识点，检索课程内容，生成一道单选题。
    """
    if not topic or not topic.strip():
        return "⚠️ 请输入一个具体的知识点，例如：'敏捷开发' 或 '项目生命周期'。"

    print(f"📝 正在为知识点 '{topic}' 生成练习题...")

    try:
        # 1. 检索素材
        docs = retriever.invoke(topic)

        # 容错处理：如果没有检索到，给一个空字符串，让 Prompt 决定怎么做
        if docs:
            context_text = "\n".join([d.page_content for d in docs[:3]])
        else:
            context_text = "（未检索到具体课程内容，请基于该知识点的通用概念出题）"

        # 2. 构建 Prompt
        # 优化策略：
        quiz_template = """
        你是一名专业的大学课程出题老师。请针对目标【知识点】出一道单项选择题。

        【参考课程内容】：
        {context}

        【目标知识点】：{topic}

        【出题要求】：
        1. 优先依据【参考课程内容】出题。如果内容中未包含具体细节（如仅有标题），请基于你对该【目标知识点】的专业知识进行补全，确保题目逻辑通顺。
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

        # 3. 调用 LLM 生成
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "topic": topic})

        # 4. 解析结果并格式化
        try:
            # 解析 LLM 返回的 JSON
            data = output_parser.parse(response.content)

            # 容错检查：确保 options 有 4 个
            opts = data.get('options', [])
            while len(opts) < 4:
                opts.append("（生成选项不足）")

            # 格式化输出
            display_text = (
                f"### 🎯 个性化练习题\n\n"
                f"**❓ 题目**: {data['question']}\n\n"
                f"**选项**:\n"
                f"A. {opts[0]}\n"
                f"B. {opts[1]}\n"
                f"C. {opts[2]}\n"
                f"D. {opts[3]}\n\n"
                f"---\n"
                f"**✅ 参考答案**: {data['answer']}\n\n"
                f"**💡 解析**: {data['explanation']}\n"
            )
            return display_text

        except Exception as parse_err:
            print(f"JSON 解析失败: {parse_err}")
            return f"⚠️ 题目生成数据解析错误，请重试。\n\n原始返回:\n{response.content}"

    except Exception as e:
        return f"❌ 系统错误: {str(e)}"


# ==============================================================================
# ------------------- 新增模块：文件上传处理 (File Upload) -------------------
# ==============================================================================

def preprocess_pdf(pdf_path: str):
    """处理 PDF 文件并返回分块文本列表"""
    print(f"正在处理 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # 清洗逻辑
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', cleaned_text)
    
    # 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(cleaned_text)
    
    return chunks, os.path.basename(pdf_path)


def preprocess_srt(srt_path: str):
    """处理 SRT 字幕文件并返回分块文本列表"""
    print(f"正在处理字幕: {srt_path}")
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text_content = []
    for line in lines:
        line = line.strip()
        # 跳过纯数字（字幕序号）
        if line.isdigit():
            continue
        # 跳过时间轴
        if '-->' in line:
            continue
        # 跳过空行
        if not line:
            continue
        text_content.append(line)
    
    # 合并为完整文本
    full_text = " ".join(text_content)
    cleaned_text = re.sub(r'\s+', ' ', full_text)
    
    # 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(cleaned_text)
    
    return chunks, os.path.basename(srt_path)


def add_documents_to_vectorstore(chunks, source_filename):
    """将新的文档分块添加到现有的向量库中"""
    global vectorstore, retriever, qa_chain
    
    # 准备文本和元数据
    texts = []
    metadatas = []
    base_name = os.path.splitext(os.path.basename(source_filename))[0]
    base_name = base_name.replace(" ", "_")
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{base_name}_{i}"
        text_with_id = f"来源ID: {chunk_id}\n{chunk}"
        texts.append(text_with_id)
        metadatas.append({"source": f"{chunk_id}.txt"})
        
        # 同时保存到 output 目录
        out_name = f"{chunk_id}.txt"
        with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(chunk)
    
    # 添加到向量库
    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"✅ 成功添加 {len(texts)} 个分块到向量库")
        
        # 更新检索器和 QA 链
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
    
    return len(texts)


def handle_file_upload(uploaded_file):
    """处理上传的文件"""
    if uploaded_file is None:
        return "⚠️ 请先选择要上传的文件（支持 PDF 或 SRT 格式）"
    
    try:
        # Gradio 4.44.1 中 File 组件返回文件对象，需要获取 name 属性
        if isinstance(uploaded_file, str):
            file_path = uploaded_file
        elif hasattr(uploaded_file, 'name'):
            file_path = uploaded_file.name
        else:
            # 兼容其他可能的返回格式
            file_path = str(uploaded_file)
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 检查文件类型
        if file_ext not in ['.pdf', '.srt']:
            return f"❌ 不支持的文件格式: {file_ext}\n\n支持格式: PDF (.pdf) 或 字幕文件 (.srt)"
        
        # 处理文件
        if file_ext == '.pdf':
            chunks, source_name = preprocess_pdf(file_path)
        else:  # .srt
            chunks, source_name = preprocess_srt(file_path)
        
        if not chunks:
            return f"⚠️ 文件处理失败：未能从 {file_name} 中提取到有效内容"
        
        # 添加到向量库
        chunk_count = add_documents_to_vectorstore(chunks, source_name)
        
        # 返回成功消息
        result = (
            f"✅ **文件上传成功！**\n\n"
            f"📄 **文件名**: {file_name}\n"
            f"📊 **处理结果**: 生成 {chunk_count} 个文本分块\n"
            f"💾 **存储位置**: {OUTPUT_DIR}\n"
            f"🔍 **向量库**: 已更新，现在可以基于此文件内容进行问答和练习生成\n\n"
            f"💡 **提示**: 你现在可以在「课程问答」或「个性化练习」标签页中使用新上传的内容了！"
        )
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"❌ **处理文件时发生错误**:\n\n```\n{str(e)}\n```\n\n**详细错误信息**:\n```\n{traceback.format_exc()}\n```"
        print(f"文件上传错误: {traceback.format_exc()}")
        return error_msg


# ------------------- 6. 测试 RAG（磁盘缓存 + 跨进程命中） -------------------
query = "解释项目阶段（Project Phase）和项目生命周期（Project Life Cycle）的概念，并区分项目开发与产品开发。"

try:
    # 1）检索调试（仅在缓存未命中时执行）

    retrieved_docs = retriever.invoke(query)
    print("\n=== 检索到的文档（MMR, k=15） ===")
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "")
        if source:
            chunk_id = os.path.splitext(source)[0]  # 得到 chunk_3
        else:
            chunk_id = f"unknown_{i}"
        preview = doc.page_content.replace("\n", " ")[:120]
        print(f"Doc {i} ({chunk_id}): {preview}...")


    start_total = time.time()

    # === 磁盘缓存 RAG 调用 ===
    answer = cached_rag(query)

    response_time = time.time() - start_total

    # 2）打印答案 + 响应时间
    print("\n" + "="*70)
    print("DeepSeek RAG 回答:")
    print(answer)
    print(f"总响应时间: {response_time:.3f} 秒")
    print("="*70)

    # 3）打印缓存状态
    normalized = query.strip().lower()
    cache_key = f"rag_v1:{hashlib.md5(normalized.encode()).hexdigest()}"
    print(f"缓存键: {cache_key[-12:]}")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"当前缓存大小: {len(rag_cache)} 条")

    # 4）可选：打印缓存命中统计
    if cache_key in rag_cache:
        print("缓存状态: 命中（第二次运行脚本将直接读取）")
    else:
        print("缓存状态: 未命中（已写入磁盘，下次运行将命中）")

except Exception as e:
    print(f"问答测试失败: {str(e)}")
    raise

# ------------------- 7. Gradio UI（异步 + 计时） -------------------
async def ask_question(question: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        start = time.time()

        # 1. 获取 LLM 的原始回答
        full_response = await loop.run_in_executor(None, lambda: qa_chain.run(question))
        elapsed = time.time() - start

        # 2. 使用【新正则表达式】提取来源 ID
        # 解释：匹配 chunk_ 开头，后面跟着 字母、数字、下划线、点、横杠 或 中文
        pattern = r"(chunk_[\w\.\-\u4e00-\u9fff]+)"
        sources = re.findall(pattern, full_response)

        # 去重并格式化来源
        unique_sources = list(set(sources))

        # 3. 构造最终显示的文本


        display_text = f"💡 **回答**:\n{full_response}\n\n"
        display_text += f"⏱️ **耗时**: {elapsed:.2f} 秒\n"

        if unique_sources:
            display_text += f"📚 **检测到的来源文件**: {', '.join(unique_sources)}"
        else:
            display_text += "⚠️ 未检测到明确的来源引用 (可能是通用知识回答)"

        return display_text

    except Exception as e:
        return f"❌ 错误: {str(e)}"

# ------------------- 7. Gradio UI (升级版：多功能面板) -------------------

# Tab 1: 课程问答界面
qa_interface = gradio.Interface(
    fn=ask_question,
    inputs=gradio.Textbox(label="💬 课程提问", placeholder="例如：敏捷开发的核心价值观是什么？", lines=2),
    outputs=gradio.Markdown(label="🤖 AI 回答"), # 使用 Markdown 渲染富文本
    allow_flagging="never",
    description="**基于 RAG 技术**：精准检索课程讲义与视频字幕，提供带溯源的专业解答。"
)

# Tab 2: 练习生成界面
quiz_interface = gradio.Interface(
    fn=generate_quiz_func,
    inputs=gradio.Textbox(label="🎯 输入知识点", placeholder="例如：Scrum 流程 / 瀑布模型 / 风险管理", lines=1),
    outputs=gradio.Markdown(label="📝 生成的练习题"),
    allow_flagging="never",
    description="**个性化练习**：输入你想复习的知识点，AI 将基于课程资料为你生成一道单选题及解析。"
)

# Tab 3: 文件上传界面
upload_interface = gradio.Interface(
    fn=handle_file_upload,
    inputs=gradio.File(
        label="📤 上传课程文件",
        file_types=[".pdf", ".srt"]
    ),
    outputs=gradio.Markdown(label="📋 处理结果"),
    allow_flagging="never",
    description="**文件上传**：上传 PDF 讲义或 SRT 字幕文件，系统将自动处理并添加到知识库中，支持后续问答和练习生成。"
)

# 主程序：使用 TabbedInterface 组合三个功能
demo = gradio.TabbedInterface(
    [qa_interface, quiz_interface, upload_interface],
    ["📚 课程问答", "✍️ 个性化练习", "📤 文件上传"],
    title="🎓 LearnMate 个性化学习伙伴 (MVP Alpha)",
    theme="soft"
)

if __name__ == "__main__":
    print("🚀 正在启动 LearnMate Web 服务...")
    demo.launch(share=True)