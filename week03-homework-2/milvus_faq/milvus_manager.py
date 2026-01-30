from pathlib import Path
from datetime import datetime
import json
from typing import List, Optional
import asyncio
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.embed_model = embedding_model

class MilvusManager:
    def __init__(
        self,
        collection_name: str,
        uri: str,
        backup_dir: str = "backups",):

        self.uri = uri
        self.collection_name = collection_name
        self.backup_dir = Path(backup_dir)
        self.dim = self._get_embedding_dim(embedding_model)

    def _get_embedding_dim(self, embed_model: HuggingFaceEmbedding) -> int:
        """
        获取嵌入模型的向量维度
        
        Returns:
            int: 向量维度，如果无法获取则返回 None
        """
        vector_dim = None
        
        try:
            # 方法1: 从 sentence_transformer 模型获取维度
            if isinstance(embed_model, HuggingFaceEmbedding):
                if hasattr(embed_model, '_model'):
                    sentence_transformer = embed_model._model
                    if hasattr(sentence_transformer, 'get_sentence_embedding_dimension'):
                        try:
                            vector_dim = sentence_transformer.get_sentence_embedding_dimension()
                        except Exception:
                            pass
            
            # 方法2: 通过实际生成嵌入向量来获取维度
            if vector_dim is None and isinstance(embed_model, HuggingFaceEmbedding):
                try:
                    test_embedding = embed_model.get_text_embedding("test")
                    vector_dim = len(test_embedding)
                except Exception:
                    pass
            
            # 方法3: 根据模型名称推断维度
            if vector_dim is None:
                model_name = getattr(embed_model, 'model_name', '')
                if model_name and "bge-m3" in str(model_name).lower():
                    vector_dim = 1024
        except Exception:
            # 异常情况下的兜底方案：根据模型名称推断
            model_name = getattr(embed_model, 'model_name', '')
            if model_name and "bge-m3" in str(model_name).lower():
                vector_dim = 1024
        
        return vector_dim

    def _create_snapshot(self) -> Optional[str]:
        """在更新前将当前集合数据导出为快照 JSON，返回快照文件路径，失败返回 None。"""

        vector_store = self._get_or_create_vector_store(overwrite=False)

        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            client = vector_store.client
            collection_name = vector_store.collection_name
            text_key = getattr(vector_store, "text_key", "text")
            doc_id_field = getattr(vector_store, "doc_id_field", "doc_id")
            embedding_field = getattr(vector_store, "embedding_field", "embedding")
            # 查询前需先 load 集合，否则 query 会失败或查不到数据
            if collection_name in client.list_collections():
                try:
                    client.load_collection(collection_name=collection_name)
                except Exception:
                    pass  # 已 load 或暂不可 load 时继续尝试 query

            # 含向量、动态字段等完整字段
            output_fields_full = [
                "id", text_key, doc_id_field, embedding_field,
                "question", "answer", "_node_content",
            ]

            # Milvus 单次 query 的 (offset+limit) 上限为 16384
            query_limit = 16384
            try:
                rows = client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=output_fields_full,
                    limit=query_limit,
                )
            except Exception:
                rows = []

            def _row_to_snapshot_item(r: dict) -> dict:
                item = {
                    "id": r.get("id"),
                    "text": r.get(text_key, ""),
                    "doc_id": r.get(doc_id_field),
                }
                if r.get(embedding_field) is not None:
                    # 向量转为 list，便于 JSON 序列化
                    emb = r.get(embedding_field)
                    item["embedding"] = emb if isinstance(emb, list) else list(emb)
                if r.get("question") is not None:
                    item["question"] = r.get("question")
                if r.get("answer") is not None:
                    item["answer"] = r.get("answer")
                if r.get("_node_content") is not None:
                    item["_node_content"] = r.get("_node_content")
                return item

            snapshot = {
                "collection_name": collection_name,
                "snapshot_at": datetime.now().isoformat(),
                "count": len(rows),
                "rows": [_row_to_snapshot_item(r) for r in rows],
            }
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{collection_name}_snapshot_{ts}.json"
            path = self.backup_dir / filename
            path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(path)
        except Exception:
            return None

    def update_collection(self, documents: List[Document]):
        self._create_snapshot()

        # 确保全局 Settings 使用本地 embed_model，避免 pipeline 内任何步骤回退到 OpenAI
        Settings.embed_model = embedding_model

        vector_store = self._get_or_create_vector_store(overwrite=True)

        splitter = SemanticSplitterNodeParser.from_defaults(
            embed_model=embedding_model
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            transformations=[splitter],
        )

    def query_collection(self):
        """查询集合"""
        vector_store = self._get_or_create_vector_store(overwrite=False)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
        )
        return index

    def _get_or_create_vector_store(self, overwrite: bool = True):
        """
        获取或创建 Milvus 向量存储
        
        Args:
            collection_name: Milvus 集合名称，默认为 "llama_collection"
            
        Returns:
            MilvusVectorStore: Milvus 向量存储对象
        """
        # 获取向量维度
        
        # 创建向量存储（不覆盖现有集合）
        # 注意：AsyncMilvusClient 需要运行中的事件循环
        try:
            # 尝试获取运行中的事件循环
            loop = asyncio.get_running_loop()
            # 如果有运行的事件循环，可以直接创建（nest_asyncio 已应用）
            vector_store = MilvusVectorStore(
                uri=self.uri,
                collection_name=self.collection_name,
                overwrite=overwrite,
                dim=self.dim,
                similarity_metric="COSINE",
            )
        except RuntimeError:
            # 如果没有运行的事件循环，创建新的事件循环并在其中创建
            async def _create_store_async():
                """在异步函数中创建向量存储（确保事件循环运行）"""
                return MilvusVectorStore(
                    uri=self.uri,
                    collection_name=self.collection_name,
                    overwrite=overwrite,
                    dim=self.dim,
                    similarity_metric="COSINE",
                )
            vector_store = asyncio.run(_create_store_async())
        
        return vector_store