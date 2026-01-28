"""
医疗行业意图识别演示案例
重点展示提示词工程能力，包含症状分析、科室推荐和紧急程度评估
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class UrgencyLevel(Enum):
    """紧急程度枚举"""
    EMERGENCY = "紧急"
    URGENT = "较急"
    NORMAL = "一般"
    NON_URGENT = "不急"

class IntentType(Enum):
    """意图类型枚举"""
    SYMPTOM_INQUIRY = "症状咨询"
    DEPARTMENT_RECOMMENDATION = "科室推荐"
    EMERGENCY_ASSESSMENT = "紧急评估"
    MEDICATION_INQUIRY = "用药咨询"
    APPOINTMENT_BOOKING = "预约挂号"
    HEALTH_EDUCATION = "健康教育"
    OTHER = "其他"

@dataclass
class MedicalResponse:
    """医疗响应数据结构"""
    intent: IntentType
    symptoms: List[str]
    recommended_department: str
    urgency_level: UrgencyLevel
    analysis: str
    suggestions: List[str]
    follow_up_questions: List[str]

class MedicalIntentRecognizer:
    """医疗意图识别器"""
    
    def __init__(self):
        self.symptom_keywords = self._load_symptom_keywords()
        self.department_mapping = self._load_department_mapping()
        self.urgency_rules = self._load_urgency_rules()
        
    def _load_symptom_keywords(self) -> Dict[str, List[str]]:
        """加载症状关键词库"""
        return {
            "头痛": ["头痛", "头疼", "偏头痛", "头晕", "头胀"],
            "发热": ["发烧", "发热", "体温高", "高烧", "低烧", "热度"],
            "咳嗽": ["咳嗽", "咳痰", "干咳", "咳血"],
            "腹痛": ["肚子疼", "腹痛", "胃痛", "肚痛"],
            "胸痛": ["胸痛", "胸闷", "心痛", "胸口疼"],
            "呼吸困难": ["呼吸困难", "气短", "喘不过气", "呼吸急促"],
            "恶心呕吐": ["恶心", "呕吐", "想吐", "反胃"],
            "皮疹": ["皮疹", "红疹", "过敏", "瘙痒", "起疹子"]
        }
    
    def _load_department_mapping(self) -> Dict[str, str]:
        """加载症状与科室映射"""
        return {
            "头痛": "神经内科",
            "发热": "内科",
            "咳嗽": "呼吸内科",
            "腹痛": "消化内科",
            "胸痛": "心内科",
            "呼吸困难": "呼吸内科",
            "恶心呕吐": "消化内科",
            "皮疹": "皮肤科"
        }
    
    def _load_urgency_rules(self) -> Dict[str, UrgencyLevel]:
        """加载紧急程度规则"""
        return {
            "胸痛": UrgencyLevel.EMERGENCY,
            "呼吸困难": UrgencyLevel.EMERGENCY,
            "高烧": UrgencyLevel.URGENT,
            "剧烈头痛": UrgencyLevel.URGENT,
            "咳血": UrgencyLevel.URGENT,
            "头痛": UrgencyLevel.NORMAL,
            "发热": UrgencyLevel.NORMAL,
            "咳嗽": UrgencyLevel.NORMAL,
            "腹痛": UrgencyLevel.NORMAL,
            "皮疹": UrgencyLevel.NON_URGENT
        }

class PromptTemplateManager:
    """提示词模板管理器"""
    
    @staticmethod
    def get_intent_classification_prompt(user_input: str) -> str:
        """意图分类提示词模板"""
        return f"""
你是一个专业的医疗AI助手，需要对患者的咨询进行意图识别和分类。

患者咨询内容："{user_input}"

请按照以下格式分析患者的意图：

1. 意图类型识别：
   - 症状咨询：患者描述身体不适症状
   - 科室推荐：患者询问应该挂哪个科室
   - 紧急评估：患者询问是否需要紧急就医
   - 用药咨询：患者询问药物使用相关问题
   - 预约挂号：患者想要预约医生或科室
   - 健康教育：患者询问疾病预防或健康知识
   - 其他：不属于以上类别的咨询

2. 症状提取：
   从患者描述中提取具体症状关键词

3. 情感分析：
   分析患者的焦虑程度和紧急感

请以JSON格式返回分析结果：
{{
    "intent_type": "意图类型",
    "extracted_symptoms": ["症状1", "症状2"],
    "emotion_level": "焦虑程度(低/中/高)",
    "urgency_indicators": ["紧急指标1", "紧急指标2"]
}}
"""

    @staticmethod
    def get_symptom_analysis_prompt(symptoms: List[str], user_context: str) -> str:
        """症状分析提示词模板"""
        symptoms_str = "、".join(symptoms)
        return f"""
作为专业医疗AI助手，请对以下症状进行详细分析：

症状列表：{symptoms_str}
患者描述："{user_context}"

请按照以下结构进行分析：

1. 症状特征分析：
   - 主要症状：识别最重要的症状
   - 伴随症状：识别次要或相关症状
   - 症状严重程度：轻微/中等/严重
   - 持续时间：急性/亚急性/慢性

2. 可能疾病方向：
   - 列出2-3个最可能的疾病方向
   - 说明判断依据

3. 科室推荐：
   - 推荐最适合的科室
   - 说明推荐理由

4. 紧急程度评估：
   - 紧急/较急/一般/不急
   - 评估依据

5. 初步建议：
   - 立即处理建议
   - 就医时间建议
   - 注意事项

请以结构化格式返回分析结果。
"""

    @staticmethod
    def get_follow_up_questions_prompt(symptoms: List[str], intent_type: str) -> str:
        """后续问题生成提示词模板"""
        symptoms_str = "、".join(symptoms)
        return f"""
基于患者的症状（{symptoms_str}）和咨询意图（{intent_type}），
生成3-5个有助于进一步诊断的后续问题。

问题设计原则：
1. 针对性强：直接关联当前症状
2. 层次清晰：从基本信息到详细特征
3. 易于回答：患者能够理解和回答
4. 诊断价值：有助于缩小诊断范围

请按重要性排序，生成后续问题列表：
1. [最重要的问题]
2. [次重要的问题]
3. [补充问题]
...

每个问题后请简要说明询问目的。
"""

    @staticmethod
    def get_emergency_assessment_prompt(symptoms: List[str], user_description: str) -> str:
        """紧急程度评估提示词模板"""
        symptoms_str = "、".join(symptoms)
        return f"""
请对以下情况进行紧急程度评估：

症状：{symptoms_str}
患者描述："{user_description}"

评估标准：
- 紧急（立即就医）：生命体征不稳定，可能危及生命
- 较急（尽快就医）：症状严重，可能快速恶化
- 一般（正常就医）：症状明显但相对稳定
- 不急（观察或择期就医）：症状轻微，不影响日常生活

危险信号识别：
- 胸痛伴呼吸困难
- 高热伴意识改变
- 剧烈头痛伴视觉异常
- 呼吸困难加重
- 大量出血
- 严重腹痛伴呕吐

请给出：
1. 紧急程度等级
2. 评估依据
3. 建议处理方式
4. 注意事项
"""

class MedicalDialogueManager:
    """医疗对话管理器"""
    
    def __init__(self):
        self.recognizer = MedicalIntentRecognizer()
        self.prompt_manager = PromptTemplateManager()
        self.conversation_history = []
        
    def process_user_input(self, user_input: str) -> MedicalResponse:
        """处理用户输入并返回医疗响应"""
        
        # 1. 意图识别
        intent = self._classify_intent(user_input)
        
        # 2. 症状提取
        symptoms = self._extract_symptoms(user_input)
        
        # 3. 科室推荐
        department = self._recommend_department(symptoms)
        
        # 4. 紧急程度评估
        urgency = self._assess_urgency(symptoms, user_input)
        
        # 5. 生成分析和建议
        analysis = self._generate_analysis(symptoms, user_input)
        suggestions = self._generate_suggestions(symptoms, urgency)
        
        # 6. 生成后续问题
        follow_up_questions = self._generate_follow_up_questions(symptoms, intent)
        
        # 7. 记录对话历史
        self.conversation_history.append({
            "user_input": user_input,
            "timestamp": "2024-01-01 12:00:00",  # 实际应用中使用真实时间戳
            "response": {
                "intent": intent.value,
                "symptoms": symptoms,
                "department": department,
                "urgency": urgency.value
            }
        })
        
        return MedicalResponse(
            intent=intent,
            symptoms=symptoms,
            recommended_department=department,
            urgency_level=urgency,
            analysis=analysis,
            suggestions=suggestions,
            follow_up_questions=follow_up_questions
        )
    
    def _classify_intent(self, user_input: str) -> IntentType:
        """分类用户意图"""
        # 关键词匹配方式的简化实现
        if any(keyword in user_input for keyword in ["疼", "痛", "不舒服", "症状"]):
            return IntentType.SYMPTOM_INQUIRY
        elif any(keyword in user_input for keyword in ["挂号", "预约", "看医生"]):
            return IntentType.APPOINTMENT_BOOKING
        elif any(keyword in user_input for keyword in ["科室", "哪个科"]):
            return IntentType.DEPARTMENT_RECOMMENDATION
        elif any(keyword in user_input for keyword in ["紧急", "急诊", "严重"]):
            return IntentType.EMERGENCY_ASSESSMENT
        elif any(keyword in user_input for keyword in ["药", "吃什么药"]):
            return IntentType.MEDICATION_INQUIRY
        else:
            return IntentType.SYMPTOM_INQUIRY  # 默认为症状咨询
    
    def _extract_symptoms(self, user_input: str) -> List[str]:
        """提取症状关键词"""
        extracted_symptoms = []
        for symptom, keywords in self.recognizer.symptom_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                extracted_symptoms.append(symptom)
        return extracted_symptoms
    
    def _recommend_department(self, symptoms: List[str]) -> str:
        """推荐科室"""
        if not symptoms:
            return "内科"
        
        # 根据主要症状推荐科室
        primary_symptom = symptoms[0]
        return self.recognizer.department_mapping.get(primary_symptom, "内科")
    
    def _assess_urgency(self, symptoms: List[str], user_input: str) -> UrgencyLevel:
        """评估紧急程度"""
        if not symptoms:
            return UrgencyLevel.NORMAL
        
        # 检查紧急关键词
        emergency_keywords = ["剧烈", "严重", "急性", "突然", "无法忍受"]
        if any(keyword in user_input for keyword in emergency_keywords):
            return UrgencyLevel.URGENT
        
        # 根据症状评估
        for symptom in symptoms:
            if symptom in self.recognizer.urgency_rules:
                return self.recognizer.urgency_rules[symptom]
        
        return UrgencyLevel.NORMAL
    
    def _generate_analysis(self, symptoms: List[str], user_input: str) -> str:
        """生成症状分析"""
        if not symptoms:
            return "未识别到明确症状，建议详细描述您的不适感受。"
        
        analysis = f"根据您描述的症状（{', '.join(symptoms)}），"
        
        if len(symptoms) == 1:
            analysis += f"主要表现为{symptoms[0]}。"
        else:
            analysis += f"主要症状为{symptoms[0]}，伴有{', '.join(symptoms[1:])}。"
        
        # 添加可能的病因分析
        if "头痛" in symptoms:
            analysis += "可能与紧张性头痛、偏头痛或其他神经系统疾病相关。"
        elif "发热" in symptoms:
            analysis += "可能提示感染性疾病，需要进一步检查确定感染源。"
        elif "咳嗽" in symptoms:
            analysis += "可能与呼吸道感染、过敏或其他肺部疾病相关。"
        
        return analysis
    
    def _generate_suggestions(self, symptoms: List[str], urgency: UrgencyLevel) -> List[str]:
        """生成建议"""
        suggestions = []
        
        # 根据紧急程度给出建议
        if urgency == UrgencyLevel.EMERGENCY:
            suggestions.append("建议立即前往急诊科就医")
            suggestions.append("如症状加重，请拨打120急救电话")
        elif urgency == UrgencyLevel.URGENT:
            suggestions.append("建议尽快就医，不要拖延")
            suggestions.append("密切观察症状变化")
        else:
            suggestions.append("建议正常时间就医")
            suggestions.append("注意休息，保持充足睡眠")
        
        # 根据症状给出具体建议
        if "发热" in symptoms:
            suggestions.append("多喝水，注意体温监测")
            suggestions.append("如体温超过38.5°C，可考虑物理降温")
        
        if "头痛" in symptoms:
            suggestions.append("避免强光刺激，保持安静环境")
            suggestions.append("可适当按摩太阳穴缓解")
        
        if "咳嗽" in symptoms:
            suggestions.append("避免吸烟和二手烟")
            suggestions.append("保持室内空气湿润")
        
        return suggestions
    
    def _generate_follow_up_questions(self, symptoms: List[str], intent: IntentType) -> List[str]:
        """生成后续问题"""
        questions = []
        
        if not symptoms:
            questions.extend([
                "请详细描述您的不适症状？",
                "症状是什么时候开始的？",
                "有什么诱发因素吗？"
            ])
            return questions
        
        # 通用问题
        questions.extend([
            "症状持续多长时间了？",
            "症状的严重程度如何（1-10分）？",
            "有什么因素会加重或缓解症状吗？"
        ])
        
        # 针对特定症状的问题
        if "头痛" in symptoms:
            questions.extend([
                "头痛的具体位置在哪里？",
                "是持续性疼痛还是阵发性疼痛？",
                "伴有恶心、呕吐或视觉异常吗？"
            ])
        
        if "发热" in symptoms:
            questions.extend([
                "最高体温是多少？",
                "发热伴有寒战吗？",
                "有其他感染症状吗（如咽痛、流涕）？"
            ])
        
        if "咳嗽" in symptoms:
            questions.extend([
                "是干咳还是有痰？",
                "痰的颜色和性质如何？",
                "咳嗽在什么时候比较严重？"
            ])
        
        return questions[:5]  # 限制问题数量

def demonstrate_medical_intent_recognition():
    """演示医疗意图识别功能"""
    
    print("=" * 60)
    print("医疗行业意图识别演示案例")
    print("=" * 60)
    
    # 创建对话管理器
    dialogue_manager = MedicalDialogueManager()
    
    # 测试用例
    test_cases = [
        "我头痛得厉害，已经持续两天了",
        "孩子发烧38.5度，还咳嗽，应该看哪个科？",
        "胸口疼，呼吸困难，这严重吗？",
        "肚子疼，恶心想吐，需要马上去医院吗？",
        "皮肤起红疹，很痒，该怎么办？"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n【测试案例 {i}】")
        print(f"患者咨询：{test_input}")
        print("-" * 40)
        
        # 处理用户输入
        response = dialogue_manager.process_user_input(test_input)
        
        # 输出分析结果
        print(f"意图类型：{response.intent.value}")
        print(f"识别症状：{', '.join(response.symptoms) if response.symptoms else '无明确症状'}")
        print(f"推荐科室：{response.recommended_department}")
        print(f"紧急程度：{response.urgency_level.value}")
        print(f"症状分析：{response.analysis}")
        
        print("\n建议措施：")
        for j, suggestion in enumerate(response.suggestions, 1):
            print(f"  {j}. {suggestion}")
        
        print("\n后续问题：")
        for j, question in enumerate(response.follow_up_questions[:3], 1):
            print(f"  {j}. {question}")
        
        print("=" * 60)

def demonstrate_prompt_optimization():
    """演示提示词优化过程"""
    
    print("\n" + "=" * 60)
    print("提示词优化演示")
    print("=" * 60)
    
    prompt_manager = PromptTemplateManager()
    
    # 示例：展示不同版本的提示词
    user_input = "我头痛得厉害，还有点发烧"
    
    print("【基础版提示词】")
    basic_prompt = f"用户说：{user_input}，请分析症状。"
    print(basic_prompt)
    
    print("\n【优化版提示词】")
    optimized_prompt = prompt_manager.get_intent_classification_prompt(user_input)
    print(optimized_prompt)
    
    print("\n【提示词优化要点】")
    optimization_points = [
        "1. 角色定义：明确AI助手的专业身份",
        "2. 任务描述：清晰说明需要完成的任务",
        "3. 输出格式：指定结构化的返回格式",
        "4. 分类标准：提供明确的分类依据",
        "5. 示例引导：通过格式示例引导输出",
        "6. 约束条件：设置必要的限制和要求"
    ]
    
    for point in optimization_points:
        print(point)

def create_extensible_template():
    """创建可扩展的提示词模板"""
    
    template_config = {
        "症状库扩展": {
            "新增症状": "在symptom_keywords中添加新的症状及其关键词",
            "示例": {
                "关节痛": ["关节痛", "关节疼", "关节炎", "风湿"],
                "失眠": ["失眠", "睡不着", "入睡困难", "早醒"]
            }
        },
        "科室映射扩展": {
            "新增科室": "在department_mapping中添加症状与科室的映射",
            "示例": {
                "关节痛": "风湿免疫科",
                "失眠": "神经内科"
            }
        },
        "紧急程度规则扩展": {
            "新增规则": "在urgency_rules中添加症状的紧急程度",
            "示例": {
                "关节痛": "UrgencyLevel.NORMAL",
                "失眠": "UrgencyLevel.NON_URGENT"
            }
        },
        "提示词模板扩展": {
            "新增模板": "在PromptTemplateManager中添加新的提示词方法",
            "命名规范": "get_[功能名称]_prompt",
            "参数设计": "根据具体需求设计输入参数",
            "输出格式": "保持一致的结构化输出格式"
        }
    }
    
    print("\n" + "=" * 60)
    print("可扩展提示词模板设计")
    print("=" * 60)
    
    for category, details in template_config.items():
        print(f"\n【{category}】")
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"{key}：")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}：{value}")

if __name__ == "__main__":
    # 运行演示
    demonstrate_medical_intent_recognition()
    demonstrate_prompt_optimization()
    create_extensible_template()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n系统特点：")
    print("✓ 结构化意图识别")
    print("✓ 多维度症状分析") 
    print("✓ 智能科室推荐")
    print("✓ 紧急程度评估")
    print("✓ 多轮对话引导")
    print("✓ 可扩展模板设计")
    print("✓ 提示词工程优化")