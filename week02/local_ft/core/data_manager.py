import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

class DataManager:
    def __init__(self):
        self.data_dir = Path("training_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.data_dir / "uploaded").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        
    def upload_dataset(self, files: List[str], dataset_name: str) -> str:
        """上传数据集文件"""
        if not files:
            return "请选择要上传的文件"
        
        if not dataset_name:
            return "请输入数据集名称"
        
        try:
            dataset_dir = self.data_dir / "uploaded" / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            uploaded_files = []
            for file_path in files:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    dest_path = dataset_dir / file_name
                    shutil.copy2(file_path, dest_path)
                    uploaded_files.append(file_name)
            
            return f"✅ 成功上传 {len(uploaded_files)} 个文件到数据集 '{dataset_name}'"
        
        except Exception as e:
            return f"❌ 上传失败: {str(e)}"
    
    def create_custom_dataset(self, instructions: str, inputs: str, outputs: str, dataset_name: str) -> str:
        """创建自定义数据集"""
        try:
            inst_list = [line.strip() for line in instructions.split('\n') if line.strip()]
            input_list = [line.strip() for line in inputs.split('\n') if line.strip()]
            output_list = [line.strip() for line in outputs.split('\n') if line.strip()]
            
            if len(inst_list) != len(input_list) or len(inst_list) != len(output_list):
                return "❌ 指令、输入和输出的行数必须相同"
            
            data = []
            for i in range(len(inst_list)):
                data.append({
                    "instruction": inst_list[i],
                    "input": input_list[i],
                    "output": output_list[i]
                })
            
            output_path = self.data_dir / "processed" / f"{dataset_name}.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            return f"✅ 成功创建包含 {len(data)} 条数据的数据集: {dataset_name}"
        
        except Exception as e:
            return f"❌ 创建失败: {str(e)}"
    
    def get_available_datasets(self) -> List[str]:
        """获取可用的数据集"""
        datasets = []
        
        # 处理后的数据集
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            for file in processed_dir.glob("*.jsonl"):
                datasets.append(f"custom:{file.stem}")
        
        # 预定义数据集
        predefined = [
            "AI-ModelScope/alpaca-gpt4-data-zh",
            "AI-ModelScope/alpaca-gpt4-data-en", 
            "swift/self-cognition",
            "AI-ModelScope/chinese-medical-dialogue",
            "AI-ModelScope/code-alpaca-zh"
        ]
        datasets.extend(predefined)
        
        return datasets
    
    def preview_dataset(self, dataset_path: str, num_samples: int = 3) -> str:
        """预览数据集"""
        try:
            if dataset_path.startswith("custom:"):
                file_name = dataset_path.replace("custom:", "")
                file_path = self.data_dir / "processed" / f"{file_name}.jsonl"
                
                if not file_path.exists():
                    return "❌ 数据集文件不存在"
                
                samples = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= num_samples:
                            break
                        if line.strip():
                            samples.append(json.loads(line))
                
                preview_text = f"📋 数据集预览 ({len(samples)} 条样本):\n\n"
                for i, sample in enumerate(samples):
                    preview_text += f"样本 {i+1}:\n"
                    preview_text += f"指令: {sample.get('instruction', 'N/A')}\n"
                    preview_text += f"输入: {sample.get('input', 'N/A')}\n"
                    preview_text += f"输出: {sample.get('output', 'N/A')}\n"
                    preview_text += "-" * 50 + "\n"
                
                return preview_text
            else:
                return f"📋 预定义数据集: {dataset_path}\n请参考官方文档了解数据格式"
        
        except Exception as e:
            return f"❌ 预览失败: {str(e)}"
    
    def delete_dataset(self, dataset_name: str) -> str:
        """删除数据集"""
        try:
            if dataset_name.startswith("custom:"):
                file_name = dataset_name.replace("custom:", "")
                file_path = self.data_dir / "processed" / f"{file_name}.jsonl"
                
                if file_path.exists():
                    file_path.unlink()
                    return f"✅ 成功删除数据集: {dataset_name}"
                else:
                    return "❌ 数据集文件不存在"
            else:
                return "❌ 无法删除预定义数据集"
        
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

# 全局实例
data_manager = DataManager()