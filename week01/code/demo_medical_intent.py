"""
医疗意图识别演示脚本
展示完整的医疗咨询处理流程
"""

from medical_intent_recognition import MedicalDialogueManager, PromptTemplateManager
import json

def run_interactive_demo():
    """运行交互式演示"""
    print("🏥 医疗意图识别系统演示")
    print("=" * 50)
    print("输入 'quit' 退出演示")
    print("=" * 50)
    
    dialogue_manager = MedicalDialogueManager()
    
    while True:
        user_input = input("\n患者咨询: ")
        
        if user_input.lower() == 'quit':
            print("感谢使用医疗意图识别系统！")
            break
        
        if not user_input.strip():
            print("请输入您的症状或问题")
            continue
        
        try:
            # 处理用户输入
            response = dialogue_manager.process_user_input(user_input)
            
            # 显示分析结果
            print("\n📋 分析结果:")
            print(f"意图类型: {response.intent.value}")
            print(f"识别症状: {', '.join(response.symptoms) if response.symptoms else '无明确症状'}")
            print(f"推荐科室: {response.recommended_department}")
            print(f"紧急程度: {response.urgency_level.value}")
            
            print(f"\n🔍 症状分析:")
            print(response.analysis)
            
            print(f"\n💡 建议措施:")
            for i, suggestion in enumerate(response.suggestions, 1):
                print(f"  {i}. {suggestion}")
            
            print(f"\n❓ 后续问题:")
            for i, question in enumerate(response.follow_up_questions[:3], 1):
                print(f"  {i}. {question}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"处理出错: {e}")

def demonstrate_prompt_engineering():
    """演示提示词工程技巧"""
    print("\n🔧 提示词工程演示")
    print("=" * 50)
    
    prompt_manager = PromptTemplateManager()
    
    # 示例用户输入
    test_input = "我头痛得厉害，还有点发烧，应该看哪个科？"
    
    print("📝 原始用户输入:")
    print(f"'{test_input}'")
    
    print("\n🎯 意图分类提示词:")
    intent_prompt = prompt_manager.get_intent_classification_prompt(test_input)
    print(intent_prompt[:300] + "..." if len(intent_prompt) > 300 else intent_prompt)
    
    print("\n🔍 症状分析提示词:")
    symptom_prompt = prompt_manager.get_symptom_analysis_prompt(["头痛", "发热"], test_input)
    print(symptom_prompt[:300] + "..." if len(symptom_prompt) > 300 else symptom_prompt)
    
    print("\n⚡ 紧急评估提示词:")
    emergency_prompt = prompt_manager.get_emergency_assessment_prompt(["头痛", "发热"], test_input)
    print(emergency_prompt[:300] + "..." if len(emergency_prompt) > 300 else emergency_prompt)

def show_system_architecture():
    """展示系统架构"""
    print("\n🏗️ 系统架构说明")
    print("=" * 50)
    
    architecture = {
        "核心组件": {
            "MedicalIntentRecognizer": "意图识别核心引擎",
            "PromptTemplateManager": "提示词模板管理器", 
            "MedicalDialogueManager": "对话管理器"
        },
        "数据结构": {
            "IntentType": "意图类型枚举",
            "UrgencyLevel": "紧急程度枚举",
            "MedicalResponse": "医疗响应数据结构"
        },
        "配置文件": {
            "medical_config.json": "症状库、科室映射、紧急规则配置"
        },
        "扩展能力": {
            "症状库扩展": "支持添加新症状和关键词",
            "科室映射扩展": "支持添加新科室",
            "提示词模板扩展": "支持自定义提示词模板"
        }
    }
    
    for category, items in architecture.items():
        print(f"\n📦 {category}:")
        for key, value in items.items():
            print(f"  • {key}: {value}")

def demonstrate_accuracy_optimization():
    """演示识别准确率优化方法"""
    print("\n📈 准确率优化演示")
    print("=" * 50)
    
    optimization_strategies = {
        "关键词优化": [
            "扩充同义词库",
            "添加方言表达",
            "包含口语化描述",
            "考虑拼写错误"
        ],
        "上下文理解": [
            "多轮对话记忆",
            "症状关联分析",
            "时间序列考虑",
            "严重程度判断"
        ],
        "提示词工程": [
            "角色定位明确",
            "任务描述详细",
            "输出格式规范",
            "示例引导充分"
        ],
        "规则优化": [
            "紧急程度细化",
            "科室映射精确",
            "异常情况处理",
            "边界条件考虑"
        ]
    }
    
    for strategy, methods in optimization_strategies.items():
        print(f"\n🎯 {strategy}:")
        for method in methods:
            print(f"  ✓ {method}")

def main():
    """主函数"""
    print("🚀 医疗意图识别系统完整演示")
    print("=" * 60)
    
    while True:
        print("\n请选择演示模式:")
        print("1. 交互式对话演示")
        print("2. 提示词工程演示") 
        print("3. 系统架构说明")
        print("4. 准确率优化演示")
        print("5. 批量测试演示")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == '0':
            print("感谢使用！")
            break
        elif choice == '1':
            run_interactive_demo()
        elif choice == '2':
            demonstrate_prompt_engineering()
        elif choice == '3':
            show_system_architecture()
        elif choice == '4':
            demonstrate_accuracy_optimization()
        elif choice == '5':
            run_batch_test()
        else:
            print("无效选择，请重新输入")

def run_batch_test():
    """运行批量测试"""
    print("\n🧪 批量测试演示")
    print("=" * 50)
    
    dialogue_manager = MedicalDialogueManager()
    
    test_cases = [
        {
            "input": "我头痛得厉害，已经持续两天了",
            "expected_intent": "症状咨询",
            "expected_department": "神经内科"
        },
        {
            "input": "孩子发烧38.5度，还咳嗽，应该看哪个科？",
            "expected_intent": "科室推荐", 
            "expected_department": "内科"
        },
        {
            "input": "胸口疼，呼吸困难，这严重吗？",
            "expected_intent": "紧急评估",
            "expected_department": "心内科"
        },
        {
            "input": "肚子疼，恶心想吐，需要马上去医院吗？",
            "expected_intent": "紧急评估",
            "expected_department": "消化内科"
        },
        {
            "input": "皮肤起红疹，很痒，该怎么办？",
            "expected_intent": "症状咨询",
            "expected_department": "皮肤科"
        }
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n【测试 {i}/{total_tests}】")
        print(f"输入: {test_case['input']}")
        
        response = dialogue_manager.process_user_input(test_case['input'])
        
        print(f"预期意图: {test_case['expected_intent']}")
        print(f"实际意图: {response.intent.value}")
        print(f"预期科室: {test_case['expected_department']}")
        print(f"实际科室: {response.recommended_department}")
        
        # 简单的准确率计算
        intent_correct = response.intent.value == test_case['expected_intent']
        department_correct = response.recommended_department == test_case['expected_department']
        
        if intent_correct and department_correct:
            correct_predictions += 1
            print("✅ 预测正确")
        else:
            print("❌ 预测有误")
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n📊 测试结果:")
    print(f"总测试数: {total_tests}")
    print(f"正确预测: {correct_predictions}")
    print(f"准确率: {accuracy:.1f}%")

if __name__ == "__main__":
    main()