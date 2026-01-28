from langchain_core.prompts import StringPromptTemplate
from pydantic import BaseModel, Field, field_validator
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
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('年龄必须在0-150之间')
        return v

class PersonInfoPromptTemplate(StringPromptTemplate):
    """人员信息提示模板类"""
    
    template_type: str = Field(default="person_analysis", description="模板类型")
    include_skills_analysis: bool = Field(default=True, description="是否包含技能分析")
    include_career_advice: bool = Field(default=False, description="是否包含职业建议")
    output_language: str = Field(default="chinese", description="输出语言")
    input_variables: List[str] = ["person_info", "analysis_type"]
    
    def format(self, **kwargs: Any) -> str:
        """格式化提示模板"""
        person_info = kwargs.get("person_info")
        analysis_type = kwargs.get("analysis_type", "basic")
        
        if not person_info:
            raise ValueError("person_info 参数不能为空")
        
        if isinstance(person_info, dict):
            try:
                person_info = PersonInfo(**person_info)
            except Exception as e:
                raise ValueError(f"person_info 格式错误: {e}")
        
        base_info = self._build_base_info_section(person_info)
        analysis_section = self._build_analysis_section(analysis_type, person_info)
        output_format = self._build_output_format_section()
        
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

class AdvancedPersonInfoPromptTemplate(PersonInfoPromptTemplate):
    """高级人员信息模板，支持更多工程化功能"""
    
    template_version: str = Field(default="1.0.0", description="模板版本")
    supported_languages: List[str] = Field(
        default=["chinese", "english"], 
        description="支持的语言"
    )
    enable_cache: bool = Field(default=True, description="启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存TTL(秒)")
    
    def format_with_validation(self, **kwargs: Any) -> str:
        """带验证的格式化方法"""
        self._validate_inputs(**kwargs)
        return super().format(**kwargs)
    
    def _validate_inputs(self, **kwargs: Any):
        """验证输入参数"""
        person_info = kwargs.get("person_info")
        analysis_type = kwargs.get("analysis_type")
        
        if not person_info:
            raise ValueError("person_info 不能为空")
        
        valid_analysis_types = ["basic", "career", "skills", "comprehensive"]
        if analysis_type not in valid_analysis_types:
            raise ValueError(f"analysis_type 必须是 {valid_analysis_types} 之一")
        
        if self.output_language not in self.supported_languages:
            raise ValueError(f"不支持的语言: {self.output_language}")
    
    def get_template_metadata(self) -> Dict[str, Any]:
        """获取模板元数据"""
        return {
            "version": self.template_version,
            "type": self.template_type,
            "supported_languages": self.supported_languages,
            "features": {
                "skills_analysis": self.include_skills_analysis,
                "career_advice": self.include_career_advice,
                "caching": self.enable_cache
            },
            "input_variables": self.input_variables
        }

def demo_custom_template():
    """演示自定义模板的使用"""
    template = PersonInfoPromptTemplate(
        include_skills_analysis=True,
        include_career_advice=True,
        output_language="chinese"
    )
    
    person_data = {
        "name": "张三",
        "age": 28,
        "occupation": "软件工程师",
        "skills": ["Python", "JavaScript", "React", "Docker"],
        "experience_years": 5,
        "location": "北京"
    }
    
    analysis_types = ["basic", "career"]
    
    for analysis_type in analysis_types:
        print("\n" + "="*30)
        print(f"分析类型: {analysis_type}")
        print("="*30)
        
        prompt = template.format(
            person_info=person_data,
            analysis_type=analysis_type
        )
        
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

def demo_advanced_template():
    """演示高级模板功能"""
    advanced_template = AdvancedPersonInfoPromptTemplate(
        template_version="2.0.0",
        include_skills_analysis=True,
        include_career_advice=True,
        enable_cache=True
    )
    
    metadata = advanced_template.get_template_metadata()
    print("模板元数据:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    
    person_data = PersonInfo(
        name="李四",
        age=32,
        occupation="产品经理",
        skills=["产品设计", "数据分析", "项目管理"],
        experience_years=8,
        location="上海"
    )
    
    try:
        prompt = advanced_template.format_with_validation(
            person_info=person_data,
            analysis_type="comprehensive"
        )
        print("\n高级模板生成的提示:")
        print("="*30)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    except ValueError as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    print("1. 基础模板演示")
    demo_custom_template()
    
    print("\n\n2. 高级模板演示")
    demo_advanced_template()