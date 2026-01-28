"""
vLLM 参数配置管理
提供预设配置和参数说明
"""
from typing import Dict, Any
from dataclasses import dataclass
import json

@dataclass
class VLLMConfig:
    """vLLM 配置数据类"""
    name: str
    description: str
    params: Dict[str, Any]

# 预设配置模板
PRESET_CONFIGS = {
    "conservative": VLLMConfig(
        name="保守型",
        description="低随机性，适合需要准确性的任务",
        params={
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 10,
            "repetition_penalty": 1.1,
            "max_tokens": 512
        }
    ),
    
    "balanced": VLLMConfig(
        name="平衡型",
        description="平衡创意和准确性，适合大多数场景",
        params={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.1,
            "max_tokens": 1024
        }
    ),
    
    "creative": VLLMConfig(
        name="创意型",
        description="高随机性，适合创意写作和头脑风暴",
        params={
            "temperature": 1.2,
            "top_p": 0.95,
            "top_k": 100,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.2,
            "max_tokens": 2048
        }
    ),
    
    "precise": VLLMConfig(
        name="精确型",
        description="极低随机性，适合代码生成和技术文档",
        params={
            "temperature": 0.01,
            "top_p": 0.7,
            "top_k": 5,
            "repetition_penalty": 1.2,
            "frequency_penalty": 0.3,
            "max_tokens": 1024
        }
    ),
    
    "diverse": VLLMConfig(
        name="多样型",
        description="强调内容多样性，减少重复",
        params={
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 80,
            "repetition_penalty": 1.3,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.4,
            "max_tokens": 1536
        }
    ),
    
    "beam_search": VLLMConfig(
        name="束搜索型",
        description="使用束搜索，适合需要高质量输出的场景",
        params={
            "temperature": 0.6,
            "use_beam_search": True,
            "best_of": 3,
            "length_penalty": 1.2,
            "early_stopping": True,
            "max_tokens": 1024
        }
    )
}

# 参数详细说明
PARAMETER_DESCRIPTIONS = {
    "temperature": {
        "description": "控制输出随机性的温度参数",
        "range": "0.0 - 2.0",
        "default": 0.7,
        "effect": "越低越确定，越高越随机",
        "use_cases": ["0.1-0.3: 代码生成", "0.7-0.9: 对话", "1.0-1.5: 创意写作"]
    },
    
    "top_p": {
        "description": "核采样参数，控制候选词汇的累积概率",
        "range": "0.0 - 1.0",
        "default": 0.9,
        "effect": "越小选择越集中，越大选择越多样",
        "use_cases": ["0.7-0.8: 精确任务", "0.9-0.95: 通用任务", "0.95-1.0: 创意任务"]
    },
    
    "top_k": {
        "description": "Top-K采样，限制候选词汇数量",
        "range": "1 - 100+",
        "default": 50,
        "effect": "越小选择越集中，越大选择越多样",
        "use_cases": ["5-20: 精确任务", "40-60: 通用任务", "80-100: 创意任务"]
    },
    
    "max_tokens": {
        "description": "最大生成token数量",
        "range": "1 - 模型上下文长度",
        "default": 512,
        "effect": "控制输出长度上限",
        "use_cases": ["128-512: 短回答", "1024-2048: 长文本", "4096+: 长文档"]
    },
    
    "repetition_penalty": {
        "description": "重复惩罚，减少重复内容",
        "range": "0.1 - 2.0",
        "default": 1.1,
        "effect": "大于1减少重复，小于1增加重复",
        "use_cases": ["1.0-1.1: 轻微惩罚", "1.1-1.3: 中等惩罚", "1.3+: 强烈惩罚"]
    },
    
    "frequency_penalty": {
        "description": "频率惩罚，基于词频降低重复",
        "range": "-2.0 - 2.0",
        "default": 0.0,
        "effect": "正值减少高频词，负值增加高频词",
        "use_cases": ["0.0-0.3: 轻微惩罚", "0.3-0.7: 中等惩罚", "0.7+: 强烈惩罚"]
    },
    
    "presence_penalty": {
        "description": "存在惩罚，基于词汇是否出现过进行惩罚",
        "range": "-2.0 - 2.0",
        "default": 0.0,
        "effect": "正值鼓励新词汇，负值鼓励重复词汇",
        "use_cases": ["0.0-0.3: 轻微鼓励", "0.3-0.7: 中等鼓励", "0.7+: 强烈鼓励"]
    },
    
    "min_p": {
        "description": "最小概率阈值，过滤低概率token",
        "range": "0.0 - 1.0",
        "default": 0.0,
        "effect": "提高生成质量，减少低质量输出",
        "use_cases": ["0.0: 不过滤", "0.01-0.05: 轻微过滤", "0.05+: 强烈过滤"]
    },
    
    "use_beam_search": {
        "description": "是否使用束搜索算法",
        "range": "True/False",
        "default": False,
        "effect": "提高输出质量但降低多样性",
        "use_cases": ["True: 高质量输出", "False: 多样性输出"]
    },
    
    "best_of": {
        "description": "生成多个候选结果并选择最佳",
        "range": "1 - 20",
        "default": 1,
        "effect": "增加计算成本但提高质量",
        "use_cases": ["1: 标准", "3-5: 高质量", "10+: 极高质量"]
    },
    
    "length_penalty": {
        "description": "长度惩罚，影响生成长度偏好",
        "range": "0.1 - 2.0",
        "default": 1.0,
        "effect": "大于1偏好长文本，小于1偏好短文本",
        "use_cases": ["0.5-0.8: 偏好短文本", "1.0: 中性", "1.2-1.5: 偏好长文本"]
    }
}

class VLLMConfigManager:
    """vLLM 配置管理器"""
    
    @staticmethod
    def get_preset_config(preset_name: str) -> Dict[str, Any]:
        """获取预设配置"""
        if preset_name not in PRESET_CONFIGS:
            available = list(PRESET_CONFIGS.keys())
            raise ValueError(f"未知预设: {preset_name}. 可用预设: {available}")
        
        return PRESET_CONFIGS[preset_name].params.copy()
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """列出所有预设配置"""
        return {name: config.description for name, config in PRESET_CONFIGS.items()}
    
    @staticmethod
    def get_parameter_info(param_name: str) -> Dict[str, Any]:
        """获取参数详细信息"""
        if param_name not in PARAMETER_DESCRIPTIONS:
            available = list(PARAMETER_DESCRIPTIONS.keys())
            raise ValueError(f"未知参数: {param_name}. 可用参数: {available}")
        
        return PARAMETER_DESCRIPTIONS[param_name].copy()
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, str]:
        """验证配置参数"""
        errors = {}
        
        # 验证 temperature
        if "temperature" in config:
            temp = config["temperature"]
            if not (0.0 <= temp <= 2.0):
                errors["temperature"] = f"必须在 0.0-2.0 范围内，当前值: {temp}"
        
        # 验证 top_p
        if "top_p" in config:
            top_p = config["top_p"]
            if not (0.0 <= top_p <= 1.0):
                errors["top_p"] = f"必须在 0.0-1.0 范围内，当前值: {top_p}"
        
        # 验证 top_k
        if "top_k" in config:
            top_k = config["top_k"]
            if not (top_k >= 1):
                errors["top_k"] = f"必须大于等于 1，当前值: {top_k}"
        
        # 验证 max_tokens
        if "max_tokens" in config:
            max_tokens = config["max_tokens"]
            if not (max_tokens >= 1):
                errors["max_tokens"] = f"必须大于等于 1，当前值: {max_tokens}"
        
        return errors
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    
    @staticmethod
    def print_config_comparison(configs: Dict[str, Dict[str, Any]]):
        """打印配置对比"""
        if not configs:
            return
        
        # 获取所有参数名
        all_params = set()
        for config in configs.values():
            all_params.update(config.keys())
        
        # 打印表头
        print(f"{'参数':<20}", end="")
        for name in configs.keys():
            print(f"{name:<15}", end="")
        print()
        print("-" * (20 + 15 * len(configs)))
        
        # 打印参数值
        for param in sorted(all_params):
            print(f"{param:<20}", end="")
            for config in configs.values():
                value = config.get(param, "N/A")
                print(f"{str(value):<15}", end="")
            print()

def demo_config_manager():
    """配置管理器演示"""
    print("=== vLLM 配置管理器演示 ===\n")
    
    manager = VLLMConfigManager()
    
    # 1. 列出预设配置
    print(" 可用预设配置:")
    presets = manager.list_presets()
    for name, desc in presets.items():
        print(f"  {name}: {desc}")
    print()
    
    # 2. 获取特定预设
    print(" 获取平衡型配置:")
    balanced_config = manager.get_preset_config("balanced")
    print(json.dumps(balanced_config, ensure_ascii=False, indent=2))
    print()
    
    # 3. 参数信息查询
    print(" temperature 参数详情:")
    temp_info = manager.get_parameter_info("temperature")
    for key, value in temp_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 4. 配置验证
    print(" 配置验证测试:")
    test_config = {"temperature": 2.5, "top_p": 0.9, "top_k": -1}
    errors = manager.validate_config(test_config)
    if errors:
        print("  发现错误:")
        for param, error in errors.items():
            print(f"    {param}: {error}")
    else:
        print("  配置有效")
    print()
    
    # 5. 配置对比
    print(" 配置对比:")
    comparison_configs = {
        "保守": manager.get_preset_config("conservative"),
        "平衡": manager.get_preset_config("balanced"),
        "创意": manager.get_preset_config("creative")
    }
    manager.print_config_comparison(comparison_configs)

if __name__ == "__main__":
    demo_config_manager()