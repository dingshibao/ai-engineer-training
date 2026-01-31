# 实验结果分析写在这里!
## 关键代码
- company_recognized.py，公司实体识别
- milvus_manager.py，向量数据库连接管理
- neo4j_manager.py，图数据库连接管理
- graph_rag_retriever.py，多跳查询逻辑，实体识别 -> 知识图谱关系提取 -> rag文档检索 -> 答案生成

## 结论
- 多跳查询中主要逻辑均由提示词来驱动，提示词的质量直接决定了最终生成答案的质量。
- LlamaIndex中提供KnowledgeGraphQueryEngine实现text-to-cypher的功能，主要是通过两段提示词实现；第一个提示词用于生成cypher语句，该提示词对于关系提取的影响极大，个人人为需要结合业务来选择提示词模版（只提取实体列表之间的关系还是提取和实体列表有直接或者间接的关系），第二个提示词是在cypher语句执行完成后获取到相关数据再结合query生成自然语言或者指定格式的数据。整体流程调用了两次llm。由于graph_query_synthesis_prompt参数中的schema是由Llamaindex进行提取，需要保证neo4j安装apoc插件或者设置特定参数跳过校验才能正常运行。

## 遇到的坑
- 环境变量的加载，由于.env文件并不是放在根目录上或者是服务启动的目录，所以需要手动设置.env文件的路径
- 经常使用Llamaindex的llm和embed model的全局设置，容易让人混淆和遗忘整体流程中哪些方法需要设置llm和embed model
- Llamaindex中使用OpenAILike调用deepseek模型时需要设置is_chat_model参数，因为deepseek正式版本并不支持complete接口，需要显式声明调用chat_completion接口