# 文件路径：api/api.py
import hashlib
import time
import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.learn_mate_core import (
    cached_rag,
    health_check,
    ingest_files,
    build_vectorstore_from_output,
    generate_quiz,
    list_cache,
    clear_cache,
    rag_cache,
)

app = FastAPI(
    title="LearnMate RAG API",
    description="Chapter 2 课程智能问答引擎",
    version="1.0.0"
)

# CORS 允许前端访问（可根据需要收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
INPUT_DIR.mkdir(exist_ok=True)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time_ms: int
    cache_hit: bool


class UploadResult(BaseModel):
    file: str
    chunks: int


class UploadResponse(BaseModel):
    results: List[UploadResult]
    total_chunks: int


class QuizRequest(BaseModel):
    topic: str


class QuizResponse(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: str


class CacheEntry(BaseModel):
    key: str
    preview: str
    length: int


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    saved_paths: List[str] = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".pdf", ".srt"]:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

        dest = INPUT_DIR / file.filename
        content = await file.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved_paths.append(str(dest))

    try:
        result_dicts = ingest_files(saved_paths)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = [UploadResult(**r) for r in result_dicts]
    total_chunks = sum(r.chunks for r in results)
    return UploadResponse(results=results, total_chunks=total_chunks)


@app.post("/api/v1/init")
async def init_vectorstore():
    try:
        count = build_vectorstore_from_output()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"vector_count": count}


@app.post("/api/v1/quiz", response_model=QuizResponse)
async def quiz(request: QuizRequest):
    try:
        result = generate_quiz(request.topic)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return QuizResponse(
        question=result["question"],
        options=result["options"],
        answer=result["answer"],
        explanation=result["explanation"],
    )


@app.get("/api/v1/history", response_model=List[CacheEntry])
async def history(limit: int = 50):
    return list_cache(limit)


@app.delete("/api/v1/cache")
async def purge_cache():
    remaining = clear_cache()
    return {"remaining": remaining}


@app.delete("/api/v1/cache/{key}")
async def delete_cache_entry(key: str):
    remaining = clear_cache(key)
    return {"remaining": remaining}


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start = time.time()
    answer = cached_rag(request.question)
    elapsed = int((time.time() - start) * 1000)

    pattern = r"(chunk_[\w\.\-\u4e00-\u9fff]+)"
    sources = list(set(re.findall(pattern, answer)))
    cache_key = f"rag_v1:{hashlib.md5(request.question.strip().lower().encode()).hexdigest()}"
    cache_hit = cache_key in rag_cache

    return AskResponse(
        answer=answer,
        sources=sources,
        response_time_ms=elapsed,
        cache_hit=cache_hit
    )


@app.get("/health")
def health():
    return health_check()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)