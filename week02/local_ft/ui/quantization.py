import gradio as gr
import subprocess
import os
from pathlib import Path
from ui.html_templates import nav_html

def quantize_model(
    model_path: str, 
    output_name: str, 
    quant_bits: int, 
    quant_method: str,
    dataset: str,
    max_length: int,
    num_samples: int
) -> str:
    """量化模型"""
    if not model_path:
        return "❌ 请输入模型路径"
    
    if not output_name:
        return "❌ 请输入输出名称"
    
    try:
        # 创建输出目录
        os.makedirs("quantized_models", exist_ok=True)
        
        # 构建量化命令
        cmd = [
            "swift", "export",
            "--model", model_path,
            "--quant_bits", str(quant_bits),
            "--quant_method", quant_method,
            "--output_dir", f"quantized_models/{output_name}",
            "--max_length", str(max_length)
        ]
        
        # 添加数据集参数
        if dataset and dataset != "无":
            cmd.extend(["--dataset", dataset])
            cmd.extend(["--num_samples", str(num_samples)])
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode == 0:
            return f"✅ 模型量化成功！\n输出路径: quantized_models/{output_name}\n\n{result.stdout}"
        else:
            return f"❌ 模型量化失败:\n{result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "❌ 量化超时，请检查模型大小和系统资源"
    except Exception as e:
        return f"❌ 量化过程中出现错误: {str(e)}"

def get_quantized_models() -> list:
    """获取已量化的模型列表"""
    quantized_dir = Path("quantized_models")
    if not quantized_dir.exists():
        return []
    
    models = []
    for model_dir in quantized_dir.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)
    
    return sorted(models, reverse=True)

def get_merged_models_for_quant() -> list:
    """获取可用于量化的合并模型"""
    merged_dir = Path("merged_models")
    models = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]  # 预定义模型
    
    if merged_dir.exists():
        for model_dir in merged_dir.iterdir():
            if model_dir.is_dir():
                models.append(f"merged_models/{model_dir.name}")
    
    return models

def delete_quantized_model(model_name: str) -> str:
    """删除量化模型"""
    if not model_name:
        return "❌ 请选择要删除的模型"
    
    try:
        model_path = Path("quantized_models") / model_name
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            return f"✅ 成功删除量化模型: {model_name}"
        else:
            return f"❌ 模型不存在: {model_name}"
    except Exception as e:
        return f"❌ 删除失败: {str(e)}"

def get_model_info(model_path: str) -> str:
    """获取模型信息"""
    if not model_path:
        return "请选择模型"
    
    try:
        path = Path("quantized_models") / model_path
        if not path.exists():
            return "模型路径不存在"
        
        # 计算模型大小
        total_size = 0
        file_count = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        
        info = f"📁 模型路径: {path}\n"
        info += f"📊 文件数量: {file_count}\n"
        info += f"💾 模型大小: {size_mb:.2f} MB ({size_gb:.2f} GB)\n"
        info += f"📅 创建时间: {path.stat().st_mtime}"
        
        return info
    except Exception as e:
        return f"获取模型信息失败: {str(e)}"

def get_quantization_block():
    """量化界面"""
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .block {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
        }
        """
    ) as demo:
        gr.HTML(nav_html)
        
        gr.Markdown("# 🗜️ 模型量化", elem_classes=["text-center"])
        
        with gr.Tab("⚡ 量化模型"):
            gr.Markdown("### 将模型量化为INT8/INT4格式以减少内存占用")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 基础配置
                    with gr.Accordion("🔧 基础配置", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=get_merged_models_for_quant(),
                            label="选择模型",
                            interactive=True,
                            value="Qwen/Qwen2.5-7B-Instruct"
                        )
                        refresh_models_btn = gr.Button("🔄 刷新模型列表")
                        
                        output_name = gr.Textbox(
                            label="输出名称",
                            placeholder="量化后的模型名称",
                            value="quantized_model"
                        )
                
                with gr.Column(scale=1):
                    # 量化参数
                    with gr.Accordion("⚙️ 量化参数", open=True):
                        quant_bits = gr.Dropdown(
                            choices=[4, 8],
                            label="量化位数",
                            value=8
                        )
                        quant_method = gr.Dropdown(
                            choices=["bnb", "gptq", "awq"],
                            label="量化方法",
                            value="bnb"
                        )
                        max_length = gr.Number(
                            label="最大长度",
                            value=2048,
                            minimum=128
                        )
                
                with gr.Column(scale=1):
                    # 校准数据集
                    with gr.Accordion("📊 校准数据集", open=True):
                        dataset = gr.Dropdown(
                            choices=["无", "AI-ModelScope/alpaca-gpt4-data-zh", "AI-ModelScope/alpaca-gpt4-data-en"],
                            label="校准数据集",
                            value="AI-ModelScope/alpaca-gpt4-data-zh"
                        )
                        num_samples = gr.Number(
                            label="样本数量",
                            value=128,
                            minimum=1
                        )
            
            quantize_btn = gr.Button("🚀 开始量化", variant="primary", size="lg")
            
            quantize_result = gr.Textbox(
                label="量化结果",
                lines=12,
                interactive=False
            )
        
        with gr.Tab("📁 管理量化模型"):
            gr.Markdown("### 管理已量化的模型")
            
            with gr.Row():
                with gr.Column(scale=2):
                    quantized_models_dropdown = gr.Dropdown(
                        choices=get_quantized_models(),
                        label="已量化的模型",
                        interactive=True
                    )
                    
                with gr.Column(scale=1):
                    refresh_quantized_btn = gr.Button("🔄 刷新列表")
                    delete_quantized_btn = gr.Button("🗑️ 删除模型", variant="stop")
            
            manage_result = gr.Textbox(
                label="操作结果",
                lines=3,
                interactive=False
            )
            
            # 模型信息
            with gr.Accordion("📊 模型信息", open=True):
                model_info = gr.Textbox(
                    label="模型详情",
                    lines=8,
                    interactive=False,
                    placeholder="选择模型查看详细信息"
                )
        
        with gr.Tab("📋 量化说明"):
            gr.Markdown("""
            ## 🎯 量化方法说明
            
            ### INT8量化 (推荐)
            - **精度损失**: 较小
            - **压缩比**: ~50%
            - **推理速度**: 中等提升
            - **适用场景**: 平衡精度和性能
            
            ### INT4量化
            - **精度损失**: 较大
            - **压缩比**: ~75%
            - **推理速度**: 显著提升
            - **适用场景**: 资源受限环境
            
            ## 🔧 量化方法对比
            
            | 方法 | 特点 | 适用模型 |
            |------|------|----------|
            | **BNB** | 简单易用，兼容性好 | 大部分模型 |
            | **GPTQ** | 高精度，需要校准数据 | Transformer模型 |
            | **AWQ** | 激活感知，精度最高 | 新版本模型 |
            
            ## 💡 使用建议
            
            1. **首次量化**: 建议使用INT8 + BNB方法
            2. **精度要求高**: 选择AWQ方法
            3. **资源受限**: 使用INT4量化
            4. **校准数据**: 选择与目标任务相关的数据集
            """)
        
        # 事件绑定
        quantize_btn.click(
            fn=quantize_model,
            inputs=[
                model_dropdown, output_name, quant_bits, 
                quant_method, dataset, max_length, num_samples
            ],
            outputs=[quantize_result]
        )
        
        refresh_models_btn.click(
            fn=get_merged_models_for_quant,
            outputs=[model_dropdown]
        )
        
        refresh_quantized_btn.click(
            fn=get_quantized_models,
            outputs=[quantized_models_dropdown]
        )
        
        delete_quantized_btn.click(
            fn=delete_quantized_model,
            inputs=[quantized_models_dropdown],
            outputs=[manage_result]
        )
        
        quantized_models_dropdown.change(
            fn=get_model_info,
            inputs=[quantized_models_dropdown],
            outputs=[model_info]
        )
    
    return demo