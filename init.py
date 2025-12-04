import os
import time

from core.learn_mate_core import preprocess_pdf, preprocess_srt, build_vectorstore, OUTPUT_DIR

# 配置数据目录
INPUT_DIR = "data"


def process_all_files():
    print(f"🚀 开始初始化 LearnMate 知识库...")
    print(f"📂 数据源目录: {os.path.abspath(INPUT_DIR)}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误：未找到数据目录 {INPUT_DIR}，请创建并放入资料。")
        return

    # 1. 扫描并处理文件
    files = os.listdir(INPUT_DIR)
    processed_count = 0
    start_time = time.time()

    for file in files:
        file_path = os.path.join(INPUT_DIR, file)

        # 处理 PDF
        if file.lower().endswith(".pdf"):
            preprocess_pdf(file_path)
            processed_count += 1

        # 处理 字幕 (.srt)
        elif file.lower().endswith(".srt"):
            preprocess_srt(file_path)
            processed_count += 1

        else:
            if not file.startswith("."):  # 忽略隐藏文件
                print(f"⚠️ 跳过不支持的文件: {file}")

    # 2. 构建向量库
    if processed_count > 0:
        print("\nUsing saved chunks to build vector store...")
        build_vectorstore()

        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"✅ LearnMate 初始化成功！(耗时 {elapsed:.2f}s)")
        print(f"📄 共处理文件数: {processed_count}")
        print("=" * 50)

        # --- 这里就是你想要添加的“下一步指引” ---
        print("\n🌐 下一步：启动 API 服务")
        print("   运行命令 -> uvicorn api.api:app --reload")
        print("\n🔗 服务启动后，请访问以下地址进行测试：")
        print("   接口文档: http://127.0.0.1:8000/docs")
        print("   Web 界面: 如果你运行了 rag_deepseek_api.py，请查看终端输出的 Gradio 地址")
        print("=" * 50 + "\n")

    else:
        print("❌ 未找到有效文件 (.pdf 或 .srt)，请检查 data/ 目录。")


if __name__ == "__main__":
    process_all_files()