# LearnMate - 个性化课程学习伙伴

**LearnMate** 是一个基于 **RAG (检索增强生成)** 技术的智能课程助手。它不仅仅是一个问答机器人，更能处理多模态课程资料（PDF 讲义 + 视频字幕），提供带精准溯源的答疑，并能主动生成个性化练习题，帮助大学生高效复习。

## ✨ 核心功能

  * **📚 多模态资料处理**：
      * 支持 **PDF 讲义** 自动提取与清洗。
      * **[New]** 支持 **.srt 视频字幕** 解析（自动去除时间轴，合并段落）。
      * 智能分块与 ID 注入，确保检索结果可精准溯源。
  * **🤖 智能问答 (Q\&A)**：
      * 基于 **DeepSeek** 大模型推理。
      * 采用 **MMR (Maximum Marginal Relevance)** 检索算法，保证答案的多样性与准确性。
      * **持久化缓存**：集成 `diskcache`，重复问题实现 **0.01秒** 极速响应。
  * **📝 个性化练习生成**：
      * **[New]** 输入知识点，AI 自动生成单项选择题。
      * 提供完整的选项（A/B/C/D）、参考答案及详细解析。
      * 基于 LangChain 结构化输出，格式稳定可靠。
  * **🖥️ 交互界面与 API**：
      * **Web UI**: 全新 Gradio 选项卡界面，支持“课程问答”与“生成练习”一键切换。
      * **REST API**: 提供标准 FastAPI 接口，自带 Swagger/OpenAPI 文档。

## 🛠️ 技术栈

  * **LLM**: DeepSeek-Chat (OpenAI Compatible)
  * **Framework**: LangChain v0.3+, LangChain-Community, FastAPI
  * **RAG Core**: ChromaDB (Vector Store), Sentence-Transformers (Embedding)
  * **Data Processing**: PyMuPDF (PDF), Regex (SRT)
  * **UI**: Gradio
  * **Cache**: DiskCache

## 🚀 快速开始

### 1\. 克隆项目

```bash
git clone https://github.com/miaojiayi123/LearnMate.git
cd LearnMate
```

### 2\. 环境配置

建议使用 Python 3.10+ 并创建虚拟环境：

```bash
# 安装依赖
pip install -r requirements.txt
```

### 3\. 配置密钥
原则上自行调用调用大模型api，但考虑到团队写作因素，统一使用env文件里的密钥



### 4\. 准备课程数据

将你的课程资料（支持多个文件）放入 `data/` 目录：

```bash
mkdir data
# 将 PDF 或 .srt 文件复制进去
cp "你的课件.pdf" data/
cp "课程视频字幕.srt" data/
```

### 5\. 初始化知识库

运行初始化脚本，自动扫描 `data/` 目录并构建向量索引：

```bash
python init.py
```

*成功后会显示：`✅ LearnMate 初始化成功！`*

### 6\. 启动应用

你可以选择启动 **Web 可视化界面** 或 **后端 API 服务**。

#### 方式一：启动可视化界面 (Gradio)

适合直接使用和演示：

```bash
python rag_deepseek_api.py
```

启动成功后，请访问：
👉 **Web 界面**: [http://127.0.0.1:7860](http://127.0.0.1:7860)

#### 方式二：启动后端 API 服务 (FastAPI)

适合前后端分离开发或查看接口定义：

```bash
uvicorn api.api:app --reload
```

启动成功后，请访问：
👉 **接口文档 (Swagger UI)**: [http://127.0.0.1:8000/docs](https://www.google.com/url?sa=E&source=gmail&q=http://127.0.0.1:8000/docs)
👉 **接口文档 (ReDoc)**: [http://127.0.0.1:8000/redoc](https://www.google.com/url?sa=E&source=gmail&q=http://127.0.0.1:8000/redoc)

## 📂 项目结构

```text
LearnMate/
├── api/
│   └── api.py            # FastAPI 接口定义
├── core/
│   └── learn_mate_core.py # 核心处理逻辑
├── docs/
│   └── 开发文档.docx         # 项目文档
├── data/                 # [输入] 存放 PDF 和 SRT 原始文件
├── output/               # [输出] 存放清洗后的 txt 分块和向量数据库
│   ├── chroma_db/        # Chroma 向量库文件
│   └── rag_cache/        # 问答缓存
    └──chunk.txt          # 清洗后的文本分块
├── scripts/              # 脚本文件
│   ├── preprocess_pdf.py         # PDF 文件预处理脚本
│   └── rag_deepseek_api.py        # Gradio Web 启动入口
├── core/
│   └── learn_mate_core.py # 核心处理逻辑
├── init.py               # 数据预处理与建库脚本
├── requirements.txt      # 项目依赖
└── .env                  # 环境变量配置
```

## 📝 开发日志

  * **v0.5.0**: 个性化练习生成引擎上线
  * **v0.4.0**: 多模态支持与RAG优化
  * **v0.3.0**: API设计与封装
  * **v0.2.0**: prompt工程与响应时间优化
  * **v0.1.0**: 文档预处理与大模型配置 

-----

*Built with ❤️ by LearnMate Team*