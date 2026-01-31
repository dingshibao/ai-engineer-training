from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
import asyncio

class MilvusManager:
    def __init__(self, uri: str, collection_name: str, overwrite: bool = True, embedding_model: HuggingFaceEmbedding = None):
        self.uri = uri
        self.collection_name = collection_name
        self.overwrite = overwrite
        self.embedding_model = embedding_model
        self.dim = self._get_embedding_dim(embedding_model)

    
    def _get_embedding_dim(self, embedding_model: HuggingFaceEmbedding) -> int:
        random_vector = embedding_model.get_query_embedding("test")
        return len(random_vector)

    def get_vector_store(self):
        """
            获取或创建 Milvus 向量存储
        """
        try:
            # 尝试获取运行中的事件循环
            loop = asyncio.get_running_loop()
            # 如果有运行的事件循环，可以直接创建（nest_asyncio 已应用）
            vector_store = MilvusVectorStore(
                uri=self.uri,
                collection_name=self.collection_name,
                overwrite=self.overwrite,
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
                    overwrite=self.overwrite,
                    dim=self.dim,
                    similarity_metric="COSINE",
                )
            vector_store = asyncio.run(_create_store_async())
            
        return vector_store
