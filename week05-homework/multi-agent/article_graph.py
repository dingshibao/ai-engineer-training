import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from execute_log import dump_execution_log
from graph_nodes import GraphNodes
from graph_state import ArticleState

EXECUTION_LOG_FILE = "execution_log.json"


def build_graph(tools: list, checkpointer=None):
    """构建文章生成图。传入 checkpointer 可将状态持久化到文件；未传入时使用 MemorySaver 以支持执行日志。"""

    nodes = GraphNodes(mcp_tools=tools)

    graph = StateGraph(ArticleState)
    graph.add_node("research_data", nodes.research_data)
    graph.add_node("write_article", nodes.write_article)
    graph.add_node("review_article", nodes.review_article)
    graph.add_node("polish_article", nodes.polish_article)
    graph.add_edge(START, "research_data")
    graph.add_edge("research_data", "write_article")
    graph.add_edge("write_article", "review_article")
    graph.add_edge("review_article","polish_article")
    graph.add_edge("polish_article", END)
    return graph.compile(checkpointer=checkpointer)

async def main():
    """在 MCP session 连接期间加载工具并执行图（session 需保持开放以便工具调用）。"""
    import os
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools


    root_dir = Path(__file__).resolve().parent.parent 
    load_dotenv(root_dir / ".env")

    log_path = root_dir / EXECUTION_LOG_FILE
    config = {"configurable": {"thread_id": "article-session"}}

    client = MultiServerMCPClient({
        "websearch": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })
    async with client.session("websearch") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools, MemorySaver())
        result = await graph.ainvoke({
            "user_input": "我想要一篇关于人工智能的文章",
            "article_style": "通俗易懂",
            "article_length": 200,
        }, config=config)
        await dump_execution_log(graph, config, log_path)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
