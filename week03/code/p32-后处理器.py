import os
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# 增加调试日志
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger("llama_index").addHandler(logging.StreamHandler(stream=sys.stdout))


Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    embed_batch_size=6,
    embed_input_length=8192
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建检索器
vector_retriever = index.as_retriever(similarity_top_k=5)

# 测试原始检索结果
print("=== 原始检索结果 ===")
nodes = vector_retriever.retrieve("怎么休事假？")
for i, node in enumerate(nodes):
    print(f"Node {i+1} (相似度: {node.score:.4f}): {node.text[:50]}...")
    print("\n")
    print("-" * 30)


# 添加相似度后处理器
from llama_index.core.postprocessor import SimilarityPostprocessor
similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.71)

# 应用后处理器
print("\n=== 应用相似度后处理器后 (cutoff=0.7) ===")
filtered_nodes = similarity_postprocessor.postprocess_nodes(nodes)
for i, node in enumerate(filtered_nodes):
    print(f"Node {i+1} (相似度: {node.score:.4f}): {node.text[:50]}...")
    print("\n")
    print("-" * 30)


print(f"\n原始 Node 数: {len(nodes)}, 过滤后 Node 数: {len(filtered_nodes)}")