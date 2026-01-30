import asyncio
from fastapi import FastAPI
from faq_retriever import FAQRetriever
from milvus_manager import MilvusManager
from pathlib import Path
import json
from llama_index.core import Document
from typing import List

# fastapi 应用
app = FastAPI()

milvus_manager = MilvusManager(collection_name="faq_data", uri="http://localhost:19530", backup_dir=Path(__file__).parent / "backups")
faq_retriever = FAQRetriever(milvus_manager)

# 查询接口：faq_retriever.query 是同步阻塞调用，放到线程池执行避免阻塞事件循环
@app.get("/query")
async def query(question: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: faq_retriever.query(question))


# 热更新向量数据库（update_collection 为同步阻塞，放入线程池执行）
@app.post("/update")
async def update():
    """初始化 FAQ 数据"""
    documents: List[Document] = []
    parent_dir = Path(__file__).parent.parent
    faq_data_path = parent_dir / "qa_pairs.json"
    with open(faq_data_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
        for item in faq_data:
            question = item["question"]
            answer = item["answer"]
            document = Document(text=question, metadata={"question": question, "answer": answer})
            documents.append(document)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: milvus_manager.update_collection(documents))
    return {"message": "向量数据库更新成功"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)