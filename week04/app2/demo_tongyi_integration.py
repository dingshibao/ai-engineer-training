#!/usr/bin/env python3
"""
通义千问集成演示脚本
展示意图识别和大模型调用的真实效果
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import graph, State, Context


def demo_workflow_with_tongyi():
    """演示使用通义千问的工作流"""
    
    print("=" * 60)
    print("电商订单处理工作流 - 通义千问集成演示")
    print("=" * 60)
    
    # 检查API密钥配置
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("  警告: 未配置 DASHSCOPE_API_KEY 环境变量")
        print("   将使用降级模式（关键词匹配 + 预设回复）")
        print("   要使用通义千问功能，请设置环境变量：")
        print("   export DASHSCOPE_API_KEY='your_api_key'")
    else:
        print(" 已配置通义千问API密钥，将使用AI功能")
    
    print()
    
    # 测试用例
    test_cases = [
        {
            "input": "我想查询我的订单状态",
            "description": "订单查询场景"
        },
        {
            "input": "我要取消刚才下的订单",
            "description": "订单取消场景"
        },
        {
            "input": "订单有问题，需要客服帮助",
            "description": "客服咨询场景"
        },
        {
            "input": "我想修改收货地址",
            "description": "订单修改场景"
        }
    ]
    
    context = Context(user_id="demo_user", session_id="demo_session")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f" 测试用例 {i}: {test_case['description']}")
        print(f"用户输入: {test_case['input']}")
        print("-" * 40)
        
        # 创建初始状态
        initial_state = State(
            user_input=test_case['input'],
            intent="",
            order_info={},
            response="",
            next_action="",
            messages=[]
        )
        
        try:
            # 运行工作流
            result = graph.invoke(initial_state, {"configurable": context})
            
            # 显示结果
            print(f" 识别意图: {result['intent']}")
            print(f" AI回复: {result['response']}")
            
            if result.get('order_info'):
                print(f" 处理结果: {result['order_info'].get('result', 'N/A')}")
            
            print(f" 消息历史: {len(result['messages'])} 条消息")
            
        except Exception as e:
            print(f" 执行失败: {e}")
        
        print("=" * 60)
        print()


def demo_intent_recognition():
    """演示意图识别功能"""
    from src.agent.services import IntentRecognitionService
    
    print(" 意图识别服务演示")
    print("-" * 30)
    
    service = IntentRecognitionService()
    
    test_inputs = [
        "我想查询订单",
        "帮我取消订单",
        "修改收货地址",
        "有问题需要客服",
        "支付失败了",
        "这个商品怎么样",
        "今天天气不错"
    ]
    
    for user_input in test_inputs:
        result = service.recognize_intent(user_input)
        print(f"输入: '{user_input}'")
        print(f"意图: {result.intent} (置信度: {result.confidence:.2f})")
        print()


def demo_tongyi_llm():
    """演示通义千问大模型服务"""
    from src.agent.services import TongyiLLMService
    
    print(" 通义千问大模型服务演示")
    print("-" * 30)
    
    service = TongyiLLMService()
    
    test_cases = [
        ("我的订单什么时候能到？", "query_order"),
        ("我想取消订单", "cancel_order"),
        ("这个产品有什么特点？", "product_inquiry")
    ]
    
    for prompt, intent in test_cases:
        print(f"用户问题: {prompt}")
        print(f"意图类型: {intent}")
        
        try:
            response = service.generate_response(prompt, intent)
            print(f"AI回复: {response.content}")
            print(f"响应时间: {response.response_time:.2f}秒")
            print(f"使用模型: {response.model}")
        except Exception as e:
            print(f"调用失败: {e}")
        
        print("-" * 30)


if __name__ == "__main__":
    print(" 启动通义千问集成演示")
    print()
    
    # 演示各个组件
    demo_intent_recognition()
    print()
    
    demo_tongyi_llm()
    print()
    
    # 演示完整工作流
    demo_workflow_with_tongyi()
    
    print(" 演示完成！")
    print()
    print(" 提示:")
    print("1. 配置 DASHSCOPE_API_KEY 环境变量以使用通义千问功能")
    print("2. 运行 'cd app2 && langgraph dev' 启动 LangGraph Studio")
    print("3. 运行 'pytest tests/test_workflow.py -v' 执行完整测试")