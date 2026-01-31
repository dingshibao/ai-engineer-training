import asyncio
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
import transformers


async def init_milvus_data(file_path: Path):
    """MilvusVectorStore 内部使用 AsyncMilvusClient，必须在已有事件循环中创建。"""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.embed_model = embed_model
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    
    # SentenceWindowNodeParser 的 sentence_splitter 需为 Callable[[str], List[str]]，不能用 SentenceSplitter；用 from_defaults() 使用默认按句切分
    # text_splitter = SentenceWindowNodeParser.from_defaults(window_size=3)
    text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=50)

    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",
        collection_name="graph_rag_companies",
        overwrite=True,
        similarity_metric="COSINE",
        dim=_get_embedding_dim(embed_model),
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex.from_documents(
        documents=documents, 
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_splitter],
    )
    print("Milvus 向量数据初始化完成。")

def _get_embedding_dim(embed_model: HuggingFaceEmbedding) -> int:
    test_embedding = embed_model.get_text_embedding("test")
    return len(test_embedding)


if __name__ == "__main__":
    asyncio.run(init_milvus_data(Path(__file__).parent.parent / "data" / "companies.txt"))