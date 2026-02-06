"""
智能客服入口
可通过 python -m smart_customer_service.main 运行
或 cd week04-homework && python -m smart_customer_service.main
"""
from langgraph.types import StateSnapshot


import json
from order_graph import build_order_graph


def main():
    graph = build_order_graph()
    config = {"configurable": {"thread_id": "main-session"}}
    result = graph.invoke({"query": "我想查询订单号为ZT202508的订单信息"}, config=config)
    print("=== 智能客服回复 ===")
    print("最终回复:", result.get("response", ""))
    print("-" * 50, "结束对话", "-" * 50)

    if result.get("order_info"):
        print("订单信息:", json.dumps(result["order_info"], ensure_ascii=False, indent=2))

    # 输出每个 checkpoint 的信息
    print("\n" + "=" * 60)
    print("Checkpoint 历史（从最新到最早）")
    print("=" * 60)
    for i, snapshot in enumerate[StateSnapshot](graph.get_state_history(config), 1):
        print(f"\n--- Checkpoint #{i} ---")
        print(f"  step: {snapshot.metadata.get('step', '-')}")
        print(f"  source: {snapshot.metadata.get('source', '-')}")
        print(f"  checkpoint_id: {snapshot.config.get('configurable', {}).get('checkpoint_id', '-')}")
        print(f"  thread_id: {snapshot.config.get('configurable', {}).get('thread_id', '-')}")
        print(f"  created_at: {snapshot.created_at}")
        print(f"  next: {snapshot.next}")
        if snapshot.values:
            print("  values:")
            for k, v in snapshot.values.items():
                if k == "messages" and v:
                    print(f"    - {k}: [{len(v)} 条消息]")
                elif k == "order_info" and isinstance(v, dict):
                    print(f"    - {k}: {json.dumps(v, ensure_ascii=False, default=str)}")
                else:
                    val_str = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                    print(f"    - {k}: {val_str}")


if __name__ == "__main__":
    main()
