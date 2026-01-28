import gradio as gr
from core.fine_tune_manager import ft_manager
from core.data_manager import data_manager
from ui.html_templates import nav_html
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re
import numpy as np
from typing import List, Tuple
import os

def parse_training_logs(logs: str) -> Tuple[List[float], List[float], List[int]]:
    """解析训练日志，提取loss和步数"""
    lines = logs.split('\n')
    steps = []
    losses = []
    eval_losses = []
    
    for line in lines:
        train_match = re.search(r"'loss':\s*([\d.]+)", line)
        if train_match:
            step_match = re.search(r'(\d+)/\d+', line)
            if step_match:
                step = int(step_match.group(1))
                loss = float(train_match.group(1))
                steps.append(step)
                losses.append(loss)
        
        eval_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
        if eval_match:
            eval_loss = float(eval_match.group(1))
            eval_losses.append(eval_loss)
    
    return losses, eval_losses, steps

def create_loss_plot(logs: str) -> str:
    """创建loss曲线图"""
    try:
        losses, eval_losses, steps = parse_training_logs(logs)
        
        if not losses:
            return None
        
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8')
        
        if losses and steps:
            plt.plot(steps, losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        
        if eval_losses:
            eval_steps = steps[-len(eval_losses):] if len(eval_losses) <= len(steps) else steps
            plt.plot(eval_steps, eval_losses, 'r-', label='Evaluation Loss', linewidth=2, alpha=0.8)
        
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = "logs/loss_plot.png"
        os.makedirs("logs", exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    except Exception as e:
        print(f"创建loss图表失败: {e}")
        return None

def start_training_wrapper(
    model, train_type, datasets_text, torch_dtype,
    num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size,
    learning_rate, gradient_accumulation_steps,
    lora_rank, lora_alpha, target_modules,
    eval_steps, save_steps, save_total_limit, logging_steps,
    max_length, warmup_ratio, dataloader_num_workers,
    cuda_visible_devices, system, model_author, model_name
):
    """启动训练"""
    datasets = [ds.strip() for ds in datasets_text.split('\n') if ds.strip()]
    
    config = {
        "model": model,
        "train_type": train_type,
        "datasets": datasets,
        "torch_dtype": torch_dtype,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "logging_steps": logging_steps,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "dataloader_num_workers": dataloader_num_workers,
        "cuda_visible_devices": cuda_visible_devices,
        "system": system,
        "model_author": model_author,
        "model_name": model_name
    }
    
    return ft_manager.start_training(config)

def update_training_status():
    """更新训练状态"""
    status = ft_manager.get_training_status()
    logs = ft_manager.get_training_logs()
    plot_path = create_loss_plot(logs)
    
    status_text = f"🔄 状态: {status['status']}\n📊 日志行数: {status['log_count']}\n⚡ 进程存活: {status['process_alive']}"
    
    return status_text, logs, plot_path

def get_fine_tune_block():
    """微调界面"""
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
        
        gr.Markdown("# 🚀 模型微调", elem_classes=["text-center"])
        
        with gr.Tab("⚙️ 微调配置"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 基础配置
                    with gr.Accordion("🔧 基础配置", open=True):
                        model = gr.Textbox(
                            label="模型名称",
                            value="Qwen/Qwen2.5-7B-Instruct",
                            placeholder="例如: Qwen/Qwen2.5-7B-Instruct"
                        )
                        train_type = gr.Dropdown(
                            choices=["lora", "full"],
                            label="训练类型",
                            value="lora"
                        )
                        torch_dtype = gr.Dropdown(
                            choices=["bfloat16", "float16", "float32"],
                            label="数据类型",
                            value="bfloat16"
                        )
                        cuda_visible_devices = gr.Textbox(
                            label="CUDA设备",
                            value="0",
                            placeholder="例如: 0 或 0,1"
                        )
                    
                    # 数据集配置
                    with gr.Accordion("📊 数据集配置", open=True):
                        datasets_text = gr.Textbox(
                            label="数据集列表 (每行一个)",
                            value="AI-ModelScope/alpaca-gpt4-data-zh#500\nAI-ModelScope/alpaca-gpt4-data-en#500",
                            lines=4,
                            placeholder="每行输入一个数据集"
                        )
                
                with gr.Column(scale=1):
                    # 训练参数
                    with gr.Accordion("🎯 训练参数", open=True):
                        with gr.Row():
                            num_train_epochs = gr.Number(label="训练轮数", value=1, minimum=1)
                            learning_rate = gr.Number(label="学习率", value=1e-4, step=1e-5)
                        with gr.Row():
                            per_device_train_batch_size = gr.Number(label="训练批次大小", value=1, minimum=1)
                            per_device_eval_batch_size = gr.Number(label="评估批次大小", value=1, minimum=1)
                        gradient_accumulation_steps = gr.Number(label="梯度累积步数", value=16, minimum=1)
                        max_length = gr.Number(label="最大长度", value=2048, minimum=128)
                    
                    # LoRA参数
                    with gr.Accordion("🔗 LoRA参数", open=True):
                        lora_rank = gr.Number(label="LoRA Rank", value=8, minimum=1)
                        lora_alpha = gr.Number(label="LoRA Alpha", value=32, minimum=1)
                        target_modules = gr.Textbox(label="目标模块", value="all-linear")
            
            with gr.Row():
                with gr.Column():
                    # 其他参数
                    with gr.Accordion("📝 其他参数", open=False):
                        eval_steps = gr.Number(label="评估步数", value=50, minimum=1)
                        save_steps = gr.Number(label="保存步数", value=50, minimum=1)
                        save_total_limit = gr.Number(label="保存总数限制", value=2, minimum=1)
                        logging_steps = gr.Number(label="日志步数", value=5, minimum=1)
                        warmup_ratio = gr.Number(label="预热比例", value=0.05, minimum=0, maximum=1)
                        dataloader_num_workers = gr.Number(label="数据加载器工作进程数", value=4, minimum=0)
                
                with gr.Column():
                    # 模型信息
                    with gr.Accordion("🏷️ 模型信息", open=True):
                        system = gr.Textbox(
                            label="系统提示",
                            value="You are a helpful assistant.",
                            lines=2
                        )
                        model_author = gr.Textbox(label="模型作者", value="YourName")
                        model_name = gr.Textbox(label="模型名称", value="CustomModel")
            
            # 控制按钮
            with gr.Row():
                start_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg")
        
        with gr.Tab("📈 训练监控"):
            with gr.Row():
                with gr.Column(scale=1):
                    status_display = gr.Textbox(
                        label="训练状态",
                        lines=5,
                        interactive=False
                    )
                    refresh_btn = gr.Button("🔄 刷新状态", variant="secondary")
                
                with gr.Column(scale=2):
                    loss_plot = gr.Image(label="Loss曲线", type="filepath")
            
            logs_display = gr.Textbox(
                label="训练日志 (最近50行)",
                lines=20,
                interactive=False,
                max_lines=50
            )
        
        # 事件绑定
        start_btn.click(
            fn=start_training_wrapper,
            inputs=[
                model, train_type, datasets_text, torch_dtype,
                num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size,
                learning_rate, gradient_accumulation_steps,
                lora_rank, lora_alpha, target_modules,
                eval_steps, save_steps, save_total_limit, logging_steps,
                max_length, warmup_ratio, dataloader_num_workers,
                cuda_visible_devices, system, model_author, model_name
            ],
            outputs=[status_display]
        )
        
        stop_btn.click(
            fn=ft_manager.stop_training,
            outputs=[status_display]
        )
        
        refresh_btn.click(
            fn=update_training_status,
            outputs=[status_display, logs_display, loss_plot]
        )
        
        # 自动刷新 - 使用新版Gradio语法
        gr.Timer(5).tick(
            fn=update_training_status,
            outputs=[status_display, logs_display, loss_plot]
        )
    
    return demo