import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from article_graph import build_graph
from execute_log import dump_execution_log

EXECUTION_LOG_FILE = "execution_log.json"

async def main():
    """在 MCP session 连接期间加载工具并执行图（session 需保持开放以便工具调用）。"""
    import os
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

if __name__ == "__main__":
    asyncio.run(main())