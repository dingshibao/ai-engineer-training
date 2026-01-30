from milvus_manager import MilvusManager

from pathlib import Path
import json
from llama_index.core import Document
from typing import List

def init_faq_data():
    """初始化 FAQ 数据"""
    backup_dir = Path(__file__).parent / "backups"
    milvus_manager = MilvusManager(collection_name="faq_data", uri="http://localhost:19530", backup_dir=backup_dir)
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
    milvus_manager.update_collection(documents)

if __name__ == "__main__":
    init_faq_data()