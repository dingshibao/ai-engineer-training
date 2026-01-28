import os
import asyncio
import time
from pathlib import Path
import json
from datetime import datetime

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore

# 配置嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    trust_remote_code=False,
)

# 配置 LLM 模型
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model=True
)


def get_embed_model_dimension():
    """
    获取嵌入模型的向量维度
    
    Returns:
        int: 向量维度，如果无法获取则返回 None
    """
    embed_model = Settings.embed_model
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

def load_data():
    """
    加载 PDF 文档数据
    
    Returns:
        list: 加载的文档列表
        
    Raises:
        FileNotFoundError: 如果 PDF 文件不存在
    """
    print("\n步骤3: 加载文档...")
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    pdf_path = project_root / "test.pdf"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    documents = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    print(f"✓ 成功加载 {len(documents)} 个文档")
    
    return documents

def get_or_create_in_memory_vector_store():
    """
    创建基于内存的向量存储
    
    Returns:
        SimpleVectorStore: 内存向量存储对象（SimpleVectorStore 是内存存储实现）
    """
    print(f"\n步骤2: 创建内存向量存储...")
    print(f"  存储类型: SimpleVectorStore")
    print(f"  说明: 数据存储在内存中，程序退出后数据会丢失")
    
    vector_store = SimpleVectorStore()
    return vector_store


def get_or_create_vector_store(collection_name="llama_collection"):
    """
    获取或创建 Milvus 向量存储
    
    Args:
        collection_name: Milvus 集合名称，默认为 "llama_collection"
        
    Returns:
        MilvusVectorStore: Milvus 向量存储对象
    """
    # 获取向量维度
    vector_dim = get_embed_model_dimension()
    print(f"\n步骤2: 创建 Milvus 向量存储...")
    print(f"  集合名称: {collection_name}")
    print(f"  向量维度: {vector_dim}")
    print(f"  连接 URI: http://localhost:19530")
    
    # 创建向量存储（不覆盖现有集合）
    # 注意：AsyncMilvusClient 需要运行中的事件循环
    try:
        # 尝试获取运行中的事件循环
        loop = asyncio.get_running_loop()
        # 如果有运行的事件循环，可以直接创建（nest_asyncio 已应用）
        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",
            collection_name=collection_name,
            overwrite=True,
            dim=vector_dim,
            similarity_metric="COSINE"
        )
    except RuntimeError:
        # 如果没有运行的事件循环，创建新的事件循环并在其中创建
        async def _create_store_async():
            """在异步函数中创建向量存储（确保事件循环运行）"""
            return MilvusVectorStore(
                uri="http://localhost:19530",
                collection_name=collection_name,
                overwrite=True,
                dim=vector_dim,
                similarity_metric="COSINE"
            )
        vector_store = asyncio.run(_create_store_async())
    
    return vector_store
    
def add_docments_by_add_method(vector_store, documents, text_spliter, collection_name):
    """
    使用 add() 方法插入文档到向量存储
    
    Args:
        vector_store: Milvus 向量存储对象
        documents: 文档列表
        text_spliter: 文本分割器
        collection_name: 集合名称
        
    Returns:
        VectorStoreIndex: 向量索引对象
        
    Raises:
        Exception: 如果数据插入失败
    """
    try:
        # 步骤1: 文档分块
        print("\n  4.1: 文档分块...")
        all_nodes = []
        for doc_idx, doc in enumerate(documents, 1):
            # 使用 SentenceSplitter 分块
            nodes = text_spliter.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
            if doc_idx % 10 == 0 or doc_idx == len(documents):
                print(f"    已处理 {doc_idx}/{len(documents)} 个文档，生成 {len(all_nodes)} 个节点")
        
        print(f"  ✓ 文档分块完成，共生成 {len(all_nodes)} 个节点")
        
        # 步骤2: 生成向量嵌入
        print("\n  4.2: 生成向量嵌入...")
        embed_model = Settings.embed_model
        batch_size = embed_model.embed_batch_size if hasattr(embed_model, 'embed_batch_size') else 32
        
        for i in range(0, len(all_nodes), batch_size):
            batch_nodes = all_nodes[i:i + batch_size]
            batch_texts = [node.get_content() for node in batch_nodes]
            
            # 使用批量方法生成 embeddings（更高效）
            try:
                batch_embeddings = embed_model._get_text_embeddings(batch_texts)
            except Exception:
                # 如果批量方法失败，使用循环方式
                batch_embeddings = [embed_model.get_text_embedding(text) for text in batch_texts]
            
            # 将 embeddings 赋值给节点
            for node, embedding in zip(batch_nodes, batch_embeddings):
                node.embedding = embedding
            
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(all_nodes):
                print(f"    已生成 {min(i + batch_size, len(all_nodes))}/{len(all_nodes)} 个向量嵌入")
        
        print(f"  ✓ 向量嵌入生成完成")
        
        # 步骤3: 使用 add() 方法插入数据到 Milvus
        print("\n  4.3: 插入数据到 Milvus...")
        print(f"    使用 vector_store.add() 方法，最后一批设置 force_flush=True 确保立即刷新")
        
        # 分批插入，避免一次性插入过多数据
        insert_batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(all_nodes), insert_batch_size):
            batch_nodes = all_nodes[i:i + insert_batch_size]
            try:
                # 最后一批才 flush，确保数据立即刷新到磁盘
                inserted_ids = vector_store.add(
                    batch_nodes,
                    force_flush=(i + insert_batch_size >= len(all_nodes))
                )
                total_inserted += len(inserted_ids)
                print(f"    已插入 {total_inserted}/{len(all_nodes)} 条数据")
            except Exception as batch_error:
                print(f"    ⚠ 批次 {i//insert_batch_size + 1} 插入失败: {str(batch_error)}")
                import traceback
                traceback.print_exc()
                raise
        
        print(f"  ✓ 数据插入完成，共插入 {total_inserted} 条数据")
        
        # 步骤4: 显式刷新所有数据到磁盘
        print("\n  4.4: 刷新数据到磁盘...")
        client = vector_store.client
        client.flush(collection_name)
        print(f"  ✓ 数据已刷新到磁盘")
        
        # 步骤5: 验证数据
        print("\n  4.5: 验证数据...")
        from pymilvus import Collection
        using_alias = client._using if hasattr(client, '_using') else 'default'
        collection = Collection(collection_name, using=using_alias)
        collection.load()  # 确保集合已加载
        num_entities = collection.num_entities
        print(f"  ✓ 验证：集合中共有 {num_entities} 条向量数据")
        
        if num_entities == 0:
            print("\n  ⚠ 警告：数据量为 0，可能的原因：")
            print("    1. 数据插入失败")
            print("    2. Milvus 服务问题")
            raise Exception("数据插入失败：集合中数据量为 0")
        elif num_entities != total_inserted:
            print(f"\n  ⚠ 警告：插入数量 ({total_inserted}) 与集合数量 ({num_entities}) 不一致")
        else:
            print(f"  ✓ 数据验证成功！插入数量与集合数量一致")
        
        # 步骤6: 加载集合到内存（使数据可搜索）
        print("\n  4.6: 加载集合到内存...")
        vector_store.client.load_collection(collection_name)
        print(f"  ✓ 集合已加载到内存（数据可搜索）")
        
        # 步骤7: 创建 VectorStoreIndex 用于查询
        print("\n  4.7: 创建 VectorStoreIndex...")
        index = VectorStoreIndex.from_vector_store(vector_store)
        print(f"  ✓ VectorStoreIndex 创建完成")
        
    except Exception as insert_error:
        print(f"\n✗ 数据插入失败: {str(insert_error)}")
        import traceback
        traceback.print_exc()
        raise
    
    return index

def add_docments_by_from_documents_method(vector_store, documents, text_spliter, collection_name):
    """
    使用 from_documents 方法插入数据
    
    Args:
        vector_store: Milvus 向量存储对象
        documents: 文档列表
        text_spliter: 文本分割器
        collection_name: 集合名称（未使用，保留以保持接口一致性）
        
    Returns:
        VectorStoreIndex: 向量索引对象
    """
    print("\n  使用 from_documents 方法...")
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        transformations=[text_spliter],
        show_progress=True
    )
    
    return index

def load_index(collection_name="llama_collection", chunk_size: int = 512, chunk_overlap: int = 64, use_memory: bool = True):
    """
    加载索引（如果集合已存在且有数据，则加载；否则从文档构建）
    
    Args:
        collection_name: Milvus 集合名称，默认为 "llama_collection"（仅在 use_memory=False 时使用）
        chunk_size: 文本分割器块大小，默认为 512
        chunk_overlap: 文本分割器块重叠大小，默认为 64
        use_memory: 是否使用内存向量存储，默认为 True。如果为 False，则使用 Milvus
    
    Returns:
        VectorStoreIndex: 向量索引对象
    """
    # 根据参数选择向量存储类型
    if use_memory:
        vector_store = get_or_create_in_memory_vector_store()
    else:
        vector_store = get_or_create_vector_store(collection_name=collection_name)
    
    # 加载文档
    documents = load_data()
    text_spliter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_spliter.split_text
    
    print(f"\n步骤4: 插入数据到向量存储...")
    print(f"  文档数量: {len(documents)}")
    print(f"  向量存储类型: {'内存存储' if use_memory else 'Milvus'}")
    
    # 选择插入方法：add() 方法 或 from_documents() 方法
    # index = add_docments_by_add_method(vector_store, documents, text_spliter, collection_name)
    index = add_docments_by_from_documents_method(vector_store, documents, text_spliter, collection_name)
    
    return index

def query_entry(query: str, chunk_size: int = 512, chunk_overlap: int = 64, collection_name: str = "llama_collection", use_memory: bool = True):
    """
    查询入口函数
    
    Args:
        query: 查询问题
        chunk_size: 文本分割器块大小，默认为 512
        chunk_overlap: 文本分割器块重叠大小，默认为 64
        collection_name: Milvus 集合名称，默认为 "llama_collection"（仅在 use_memory=False 时使用）
        use_memory: 是否使用内存向量存储，默认为 True。如果为 False，则使用 Milvus
        
    Returns:
        dict: 包含以下键的字典：
            - response: 模型查询结果（Response 对象）
            - similarity_list: 相似度列表，每个元素包含：
                - index: 片段索引
                - score: 相似度分数
                - content: 文档片段内容（前200字符）
                - full_content: 完整文档片段内容
    """
    # 创建索引
    index = load_index(collection_name=collection_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_memory=use_memory)
    
    # ========== 查询阶段 ==========
    # 使用 ChatPromptTemplate 区分系统提示词和用户提示词（推荐）
    # 系统提示词：定义助手的行为和角色
    # 用户提示词：包含上下文和问题
    
    # 定义 refine 提示词模板（用于多步骤优化答案）
    custom_refine_prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="""你是一个专业的文档问答助手。你的任务是优化已有的回答，使其更加准确和完整。
                上下文信息：{context_str}"""
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="""用户问题：{query_str}, 请基于上下文信息回答问题。"""
            )
        ]
    )
    
    # 创建响应合成器，使用自定义提示词
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.core.response_synthesizers import get_response_synthesizer
    
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,  # 或使用 ResponseMode.REFINE
        refine_template=custom_refine_prompt,
    )
    
    # 创建查询引擎（当前已注释，可根据需要启用）
    query_engine = index.as_query_engine(
        response_synthesizer=response_synthesizer,
        similarity_top_k=5,  # 检索前5个最相关的文档块
    )
    
    # 执行查询
    print("=" * 50)
    print(f"查询问题：{query}")
    print("=" * 50)
    response = query_engine.query(query)
    print(response)
    
    # 收集相似度列表
    similarity_list = []
    print("\n" + "=" * 50)
    print("检索到的相关文档片段：")
    print("=" * 50)
    for i, node in enumerate(response.source_nodes, 1):
        similarity_info = {
            "index": i,
            "score": node.score,
            "content": node.get_content()[:200] + "...",
            "full_content": node.get_content()
        }
        similarity_list.append(similarity_info)
        print(f"\n片段 {i} (相似度: {node.score:.4f}):")
        print(node.get_content()[:200] + "...")
    
    # 返回查询结果和相似度列表
    return {
        "response": response,
        "similarity_list": similarity_list
    }


def load_results_from_json(json_file="query_results.json"):
    """
    从 JSON 文件加载查询结果，转换为 result_list 和 value_list
    
    Args:
        json_file: JSON 文件名，默认为 "query_results.json"
        
    Returns:
        tuple: (result_list, value_list)
            - result_list: 查询结果列表，格式与 query_entry 返回的格式一致
            - value_list: 查询参数列表
    """
    current_dir = Path(__file__).parent
    json_path = current_dir / json_file
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    result_list = []
    value_list = []
    
    for item in json_data:
        # 构建 value_list 项
        value_item = {
            "query": item["query"],
            "chunk_size": item["chunk_size"],
            "chunk_overlap": item["chunk_overlap"]
        }
        value_list.append(value_item)
        
        # 构建 result_list 项
        # 创建一个简单的 Response 包装类，使其可以像 Response 对象一样被 str() 转换
        class ResponseWrapper:
            def __init__(self, text):
                self.text = text
            
            def __str__(self):
                return self.text
        
        result_item = {
            "response": ResponseWrapper(item["model_response"]),
            "similarity_list": item["similarity_list"]
        }
        result_list.append(result_item)
    
    print(f"✓ 从 {json_path} 加载了 {len(result_list)} 条查询结果")
    
    return result_list, value_list


def save_results_to_table(result_list, value_list, output_file="query_results.md"):
    """
    将查询结果保存为表格格式
    
    Args:
        result_list: 查询结果列表
        value_list: 查询参数列表
        output_file: 输出文件名，默认为 "query_results.md"
    """
    current_dir = Path(__file__).parent
    output_path = current_dir / output_file
    
    # 生成 Markdown 表格
    markdown_content = []
    markdown_content.append("# 查询结果汇总表\n")
    markdown_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    markdown_content.append("\n")
    
    # ========== 汇总对比表格 ==========
    markdown_content.append("## 相似度结果对比表\n")
    markdown_content.append("\n")
    
    # 找到最大的相似度片段数量
    max_similarity_count = max(len(result["similarity_list"]) for result in result_list)
    
    # 构建表头
    header = ["配置 (chunk_size, chunk_overlap)"]
    for i in range(1, max_similarity_count + 1):
        header.append(f"Top {i} 相似度")
    header.append("模型输出")
    
    # 写入表头
    markdown_content.append("| " + " | ".join(header) + " |\n")
    markdown_content.append("|" + "|".join(["---"] * len(header)) + "|\n")
    
    # 写入数据行
    for result, value in zip(result_list, value_list):
        row = [f"{value['chunk_size']}, {value['chunk_overlap']}"]
        
        # 添加相似度分数
        similarity_list = result["similarity_list"]
        for i in range(max_similarity_count):
            if i < len(similarity_list):
                score = similarity_list[i]["score"]
                row.append(f"{score:.4f}")
            else:
                row.append("-")
        
        # 添加模型输出（截断显示）
        model_response = str(result["response"])
        # 如果模型输出太长，截断并添加省略号
        if len(model_response) > 200:
            model_response_display = model_response[:200] + "..."
        else:
            model_response_display = model_response
        # 转义表格特殊字符
        model_response_display = model_response_display.replace("|", "\\|").replace("\n", " ")
        row.append(model_response_display)
        
        markdown_content.append("| " + " | ".join(row) + " |\n")
    
    markdown_content.append("\n")
    
    # ========== 详细相似度片段对比表 ==========
    markdown_content.append("## 相似度片段详细对比表\n")
    markdown_content.append("\n")
    
    # 构建详细对比表头（包含片段内容）
    detail_header = ["配置 (chunk_size, chunk_overlap)"]
    for i in range(1, max_similarity_count + 1):
        detail_header.append(f"Top {i} 相似度")
        detail_header.append(f"Top {i} 片段内容（前100字符）")
    detail_header.append("模型输出")
    
    # 写入详细表头
    markdown_content.append("| " + " | ".join(detail_header) + " |\n")
    markdown_content.append("|" + "|".join(["---"] * len(detail_header)) + "|\n")
    
    # 写入详细数据行
    for result, value in zip(result_list, value_list):
        detail_row = [f"{value['chunk_size']}, {value['chunk_overlap']}"]
        
        # 添加相似度分数和片段内容
        similarity_list = result["similarity_list"]
        for i in range(max_similarity_count):
            if i < len(similarity_list):
                score = similarity_list[i]["score"]
                content = similarity_list[i]["content"]
                # 截断内容
                if len(content) > 100:
                    content_display = content[:100] + "..."
                else:
                    content_display = content
                # 转义表格特殊字符
                content_display = content_display.replace("|", "\\|").replace("\n", " ")
                detail_row.append(f"{score:.4f}")
                detail_row.append(content_display)
            else:
                detail_row.append("-")
                detail_row.append("-")
        
        # 添加模型输出
        model_response = str(result["response"])
        if len(model_response) > 200:
            model_response_display = model_response[:200] + "..."
        else:
            model_response_display = model_response
        model_response_display = model_response_display.replace("|", "\\|").replace("\n", " ")
        detail_row.append(model_response_display)
        
        markdown_content.append("| " + " | ".join(detail_row) + " |\n")
    
    markdown_content.append("\n")
    markdown_content.append("---\n")
    markdown_content.append("\n")
    
    # ========== 单个查询详细结果 ==========
    for idx, (result, value) in enumerate(zip(result_list, value_list), 1):
        markdown_content.append(f"## 查询 {idx}\n")
        markdown_content.append(f"**查询问题**: {value['query']}\n")
        markdown_content.append(f"**chunk_size**: {value['chunk_size']}\n")
        markdown_content.append(f"**chunk_overlap**: {value['chunk_overlap']}\n")
        markdown_content.append("\n")
        
        # 模型结果
        model_response = str(result["response"])
        markdown_content.append("### 模型回答\n")
        markdown_content.append(f"{model_response}\n")
        markdown_content.append("\n")
        
        # 相似度片段表格
        markdown_content.append("### 相似度片段\n")
        markdown_content.append("| 序号 | 相似度 | 片段内容（前200字符） |\n")
        markdown_content.append("|------|--------|----------------------|\n")
        
        for sim_item in result["similarity_list"]:
            index = sim_item["index"]
            score = sim_item["score"]
            content_preview = sim_item["content"].replace("|", "\\|").replace("\n", " ")  # 转义表格特殊字符
            markdown_content.append(f"| {index} | {score:.4f} | {content_preview} |\n")
        
        # 详细片段内容（折叠显示）
        markdown_content.append("\n<details>\n<summary>查看完整片段内容</summary>\n\n")
        for sim_item in result["similarity_list"]:
            index = sim_item["index"]
            score = sim_item["score"]
            full_content = sim_item["full_content"]
            markdown_content.append(f"#### 片段 {index} (相似度: {score:.4f})\n\n")
            markdown_content.append(f"{full_content}\n\n")
        markdown_content.append("</details>\n\n")
        
        markdown_content.append("---\n")
        markdown_content.append("\n")
    
    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(markdown_content))
    
    print(f"\n✓ 结果已保存到: {output_path}")
    
    # 同时保存为 JSON 格式（便于程序读取）
    json_output_path = current_dir / "query_results.json"
    json_data = []
    for result, value in zip(result_list, value_list):
        json_data.append({
            "query": value["query"],
            "chunk_size": value["chunk_size"],
            "chunk_overlap": value["chunk_overlap"],
            "model_response": str(result["response"]),
            "similarity_list": result["similarity_list"]
        })
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON 格式结果已保存到: {json_output_path}")


if __name__ == "__main__":
    # ========== 使用方式 ==========
    # 方式1: 从 JSON 文件加载结果（跳过查询，直接使用已保存的结果）
    # 设置 load_from_json = True，并确保 query_results.json 文件存在
    #
    # 方式2: 执行新的查询
    # 设置 load_from_json = False，程序会执行查询并保存结果
    
    load_from_json = False
    
    current_dir = Path(__file__).parent
    json_file = current_dir / "query_results.json"
    
    if load_from_json and json_file.exists():
        # 从 JSON 文件加载结果
        print("=" * 50)
        print("从 JSON 文件加载查询结果...")
        print("=" * 50)
        result_list, value_list = load_results_from_json(json_file="query_results.json")
    else:
        # 执行新的查询
        result_list = []
        
        # 测试不同 chunk_size 和 chunk_overlap 参数的效果
        # use_memory=True 表示使用内存向量存储（默认），设置为 False 则使用 Milvus
        value_list = [
            {"query": "6G网络遇到的挑战", "chunk_size": 64, "chunk_overlap": 64},
            {"query": "6G网络遇到的挑战", "chunk_size": 128, "chunk_overlap": 64},
            {"query": "6G网络遇到的挑战", "chunk_size": 256, "chunk_overlap": 64},
            {"query": "6G网络遇到的挑战", "chunk_size": 512, "chunk_overlap": 64},
            {"query": "6G网络遇到的挑战", "chunk_size": 512, "chunk_overlap": 0}
        ]
        
        for value in value_list:
            result = query_entry(
                query=value["query"],
                chunk_size=value["chunk_size"],
                chunk_overlap=value["chunk_overlap"],
                use_memory=True  # 使用内存向量存储
            )
            result_list.append(result)
    
    # 保存结果到表格文件
    save_results_to_table(result_list, value_list, output_file="query_results.md")
    
    print()