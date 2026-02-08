
from langgraph.graph import END, START, StateGraph

from graph_nodes import GraphNodes
from graph_state import ArticleState



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