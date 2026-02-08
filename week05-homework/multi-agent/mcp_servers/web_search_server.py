import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from tavily import TavilyClient

# 加载 week05-homework/.env（MCP 服务独立进程需显式加载）
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

mcp = FastMCP("WebSearchServer")


@mcp.tool()
def web_search(query: str, search_depth: str = "basic") -> str:
    """
    使用Tavily API进行网络搜索。

    Args:
        query: 搜索查询字符串。
        search_depth: 搜索深度，可选值为 "basic" 或 "advanced"。

    Returns:
        搜索结果的 JSON 字符串或错误描述。
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return json.dumps(
                {"success": False, "error": "TAVILY_API_KEY 未设置，请在 .env 中配置"},
                ensure_ascii=False,
            )
        tavily_client = TavilyClient(api_key=api_key)
        if search_depth == "basic":
            response = tavily_client.search(query, search_depth, max_results=1)
        elif search_depth == "advanced":
            response = tavily_client.search(query, search_depth, include_answer=True, max_results=1)
        else:
            return json.dumps(
                {"success": False, "error": "Invalid search depth"},
                ensure_ascii=False,
            )
        result = {
            "success": True,
            "results": response.get("results", []),
            "answer": response.get("answer"),
            "query": query,
        }
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps(
            {"success": False, "results": [], "query": query, "error": str(e)},
            ensure_ascii=False,
        )

# ASGI 应用，供 uvicorn 单独启动
app = mcp.http_app()

if __name__ == "__main__":
    import uvicorn
    # 方式1: cd week05-homework && uv run python run_mcp_server.py
    # 方式2: cd week05-homework/multi-agent && uv run python -m mcp_servers.web_search_server
    # 方式3: MCP_PORT=8001 uv run uvicorn mcp_servers.web_search_server:app --host 0.0.0.0 --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8000)