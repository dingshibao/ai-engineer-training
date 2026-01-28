import gradio as gr
import subprocess
import os
from pathlib import Path
from core.fine_tune_manager import ft_manager
from ui.html_templates import nav_html

def merge_lora_weights(checkpoint_path: str, output_name: str, merge_dtype: str) -> str:
    """合并LoRA权重"""
    if not checkpoint_path:
        return "❌ 请选择checkpoint路径"
    
    if not output_name:
        return "❌ 请输入输出模型名称"
    
    try:
        # 创建输出目录
        os.makedirs("merged_models", exist_ok=True)
        
        # 构建merge命令
        cmd = [
            "swift", "export",
            "--ckpt_dir", checkpoint_path,
            "--merge_lora", "true",
            "--output_dir", f"merged_models/{output_name}",
            "--dtype", merge_dtype
        ]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            return f"✅ 模型合并成功！\n输出路径: merged_models/{output_name}\n\n{result.stdout}"
        else:
            return f"❌ 模型合并失败:\n{result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "❌ 合并超时，请检查模型大小和系统资源"
    except Exception as e:
        return f"❌ 合并过程中出现错误: {str(e)}"

def get_merged_models() -> list:
    """获取已合并的模型列表"""
    merged_dir = Path("merged_models")
    if not merged_dir.exists():
        return []
    
    models = []
    for model_dir in merged_dir.iterdir():
        if model_dir.is_dir():
            models.append(model_dir.name)
    
    return sorted(models, reverse=True)

def delete_merged_model(model_name: str) -> str:
    """删除合并的模型"""
    if not model_name:
        return "❌ 请选择要删除的模型"
    
    try:
        model_path = Path("merged_models") / model_name
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            return f"✅ 成功删除模型: {model_name}"
        else:
            return f"❌ 模型不存在: {model_name}"
    except Exception as e:
        return f"❌ 删除失败: {str(e)}"

def get_model_merge_block():
    """模型合并界面"""
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
        
        gr.Markdown("# 🔗 LoRA权重合并", elem_classes=["text-center"])
        
        with gr.Tab("🔄 合并权重"):
            gr.Markdown("### 将LoRA权重合并到基础模型中")
            
            with gr.Row():
                with gr.Column(scale=2):
                    checkpoint_dropdown = gr.Dropdown(
                        choices=ft_manager.get_available_checkpoints(),
                        label="选择Checkpoint",
                        interactive=True
                    )
                    refresh_checkpoints_btn = gr.Button("🔄 刷新Checkpoint列表")
                    
                with gr.Column(scale=1):
                    output_name = gr.Textbox(
                        label="输出模型名称",
                        placeholder="输入合并后的模型名称",
                        value="merged_model"
                    )
                    merge_dtype = gr.Dropdown(
                        choices=["bfloat16", "float16", "float32"],
                        label="合并精度",
                        value="bfloat16"
                    )
                    merge_btn = gr.Button("🚀 开始合并", variant="primary", size="lg")
            
            merge_result = gr.Textbox(
                label="合并结果",
                lines=10,
                interactive=False
            )
        
        with gr.Tab("📁 管理模型"):
            gr.Markdown("### 管理已合并的模型")
            
            with gr.Row():
                with gr.Column(scale=2):
                    merged_models_dropdown = gr.Dropdown(
                        choices=get_merged_models(),
                        label="已合并的模型",
                        interactive=True
                    )
                    
                with gr.Column(scale=1):
                    refresh_models_btn = gr.Button("🔄 刷新模型列表")
                    delete_model_btn = gr.Button("🗑️ 删除模型", variant="stop")
            
            manage_result = gr.Textbox(
                label="操作结果",
                lines=3,
                interactive=False
            )
            
            # 模型信息展示
            with gr.Accordion("📊 模型信息", open=False):
                model_info = gr.Textbox(
                    label="模型详情",
                    lines=8,
                    interactive=False,
                    placeholder="选择模型查看详细信息"
                )
        
        # 事件绑定
        merge_btn.click(
            fn=merge_lora_weights,
            inputs=[checkpoint_dropdown, output_name, merge_dtype],
            outputs=[merge_result]
        )
        
        refresh_checkpoints_btn.click(
            fn=ft_manager.get_available_checkpoints,
            outputs=[checkpoint_dropdown]
        )
        
        refresh_models_btn.click(
            fn=get_merged_models,
            outputs=[merged_models_dropdown]
        )
        
        delete_model_btn.click(
            fn=delete_merged_model,
            inputs=[merged_models_dropdown],
            outputs=[manage_result]
        )
    
    return demo