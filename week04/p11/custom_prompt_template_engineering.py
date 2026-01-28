from langchain_core.prompts import StringPromptTemplate
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class PersonInfo(BaseModel):
    """人员信息数据模型"""
    name: str = Field(..., description="姓名")
    age: int = Field(..., ge=0, le=150, description="年龄")
    occupation: str = Field(..., description="职业")
    skills: List[str] = Field(default_factory=list, description="技能列表")
    experience_years: int = Field(default=0, ge=0, description="工作经验年数")
    location: Optional[str] = Field(None, description="所在地")
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('年龄必须在0-150之间')
        return v

class PersonInfoPromptTemplate(StringPromptTemplate):
    """
    人员信息提示模板类
    用于生成结构化的人员信息查询和分析提示
    """
    
    # 模板参数定义
    template_type: str = Field(default="person_analysis", description="模板类型")
    include_skills_analysis: bool = Field(default=True, description="是否包含技能分析")
    include_career_advice: bool = Field(default=False, description="是否包含职业建议")
    output_language: str = Field(default="chinese", description="输出语言")
    
    # 必需的输入变量
    input_variables: List[str] = ["person_info", "analysis_type"]
    
    def format(self, **kwargs: Any) -> str:
        """
        格式化提示模板
        """
        # 获取输入参数
        person_info = kwargs.get("person_info")
        analysis_type = kwargs.get("analysis_type", "basic")
        
        # 验证输入
        if not person_info:
            raise ValueError("person_info 参数不能为空")
        
        # 如果传入的是字典，转换为 PersonInfo 对象
        if isinstance(person_info, dict):
            try:
                person_info = PersonInfo(**person_info)
            except Exception as e:
                raise ValueError(f"person_info 格式错误: {e}")
        
        # 构建基础信息部分
        base_info = self._build_base_info_section(person_info)
        
        # 构建分析要求部分
        analysis_section = self._build_analysis_section(analysis_type, person_info)
        
        # 构建输出格式要求
        output_format = self._build_output_format_section()
        
        # 组合完整提示
        full_prompt = f"""# 人员信息分析任务

            ## 基础信息
            {base_info}

            ## 分析要求
            {analysis_section}

            ## 输出格式要求
            {output_format}

            请基于以上信息进行专业分析，确保分析结果准确、有用且符合格式要求。
            """
        
        return full_prompt
    
    def _build_base_info_section(self, person_info: PersonInfo) -> str:
        """构建基础信息部分"""
        info_lines = [
            f"- 姓名: {person_info.name}",
            f"- 年龄: {person_info.age}岁",
            f"- 职业: {person_info.occupation}",
            f"- 工作经验: {person_info.experience_years}年"
        ]
        
        if person_info.location:
            info_lines.append(f"- 所在地: {person_info.location}")
        
        if person_info.skills:
            skills_str = ", ".join(person_info.skills)
            info_lines.append(f"- 技能: {skills_str}")
        
        return "\n".join(info_lines)
    
    def _build_analysis_section(self, analysis_type: str, person_info: PersonInfo) -> str:
        """构建分析要求部分"""
        base_requirements = [
            "1. 对该人员的基本情况进行客观分析",
            "2. 评估其专业背景和经验水平"
        ]
        
        if analysis_type == "career":
            base_requirements.extend([
                "3. 分析职业发展轨迹和潜力",
                "4. 提供职业发展建议"
            ])
        elif analysis_type == "skills":
            base_requirements.extend([
                "3. 深入分析技能结构和优势",
                "4. 识别技能缺口和提升方向"
            ])
        elif analysis_type == "comprehensive":
            base_requirements.extend([
                "3. 综合评估职业竞争力",
                "4. 提供全面的发展建议",
                "5. 分析市场匹配度"
            ])
        
        if self.include_skills_analysis and person_info.skills:
            base_requirements.append("6. 详细分析技能组合的市场价值")
        
        if self.include_career_advice:
            base_requirements.append("7. 提供具体的职业规划建议")
        
        return "\n".join(base_requirements)
    
    def _build_output_format_section(self) -> str:
        """构建输出格式要求"""
        format_requirements = [
            "请按以下结构输出分析结果:",
            "",
            "### 基本评估",
            "- [基本情况总结]",
            "",
            "### 优势分析", 
            "- [主要优势点]",
            "",
            "### 发展建议",
            "- [具体建议]"
        ]
        
        if self.include_skills_analysis:
            format_requirements.extend([
                "",
                "### 技能分析",
                "- [技能评估和建议]"
            ])
        
        return "\n".join(format_requirements)
    
    @property
    def _prompt_type(self) -> str:
        """返回提示类型"""
        return "person_info_analysis"
    
    def save_template_config(self, filepath: str):
        """保存模板配置到文件"""
        config = {
            "template_type": self.template_type,
            "include_skills_analysis": self.include_skills_analysis,
            "include_career_advice": self.include_career_advice,
            "output_language": self.output_language,
            "created_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_template_config(cls, filepath: str) -> 'PersonInfoPromptTemplate':
        """从文件加载模板配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 移除时间戳字段
        config.pop('created_at', None)
        
        return cls(**config)
