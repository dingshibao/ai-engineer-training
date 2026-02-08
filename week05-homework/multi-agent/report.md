

# 关键代码分析
- main.py，主程序入口
- article_graph.py，构建图
- agent_define.py，智能体定义，包括Research/Write/Review/Polish
- graph_nodes.py，图节点执行方法定义
- graph_state.py，图状态定义
- execute_log.py，通过checkpointer获取执行结果输出并生成文件
- web_search_server.py，远程MCP服务，用于网络搜索

# 程序运行
- 使用uvicorn独立运行web_search_server中的MCP服务
- 使用python命令执行main.py代码即可执行生成文章的逻辑