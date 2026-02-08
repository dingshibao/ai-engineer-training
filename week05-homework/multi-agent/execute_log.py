import json
from pathlib import Path

def to_json_serializable(obj, max_len: int = 500):
    """将任意对象转为可 JSON 序列化的形式。"""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[:max_len] + "..."
    if hasattr(obj, "content") and hasattr(obj, "__class__"):
        # LangChain BaseMessage (AIMessage, HumanMessage, etc.)
        content = getattr(obj, "content", "")
        return {"_type": obj.__class__.__name__, "content": to_json_serializable(content)}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj[:20]]  # 限制长度
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in list(obj.items())[:50]}
    return str(obj)[:max_len]


def snapshot_to_dict(snapshot) -> dict:
    """将 StateSnapshot 转为可 JSON 序列化的 dict。"""
    values = {}
    if snapshot.values:
        for k, v in snapshot.values.items():
            values[k] = to_json_serializable(v)
    return {
        "step": snapshot.metadata.get("step"),
        "source": snapshot.metadata.get("source"),
        "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
        "thread_id": snapshot.config.get("configurable", {}).get("thread_id"),
        "created_at": str(snapshot.created_at) if snapshot.created_at else None,
        "next": list(snapshot.next) if snapshot.next else [],
        "values": values,
    }


async def dump_execution_log(graph, config: dict, log_path: Path):
    """将节点执行顺序和相关信息写入 JSON 文件。"""
    log_entries = []
    async for snapshot in graph.aget_state_history(config):
        log_entries.append(snapshot_to_dict(snapshot))
    # 从最早到最新排序（history 通常是最新在前）
    log_entries.reverse()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(log_entries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"执行日志已写入: {log_path}")
