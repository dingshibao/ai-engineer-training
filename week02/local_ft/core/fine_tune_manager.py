import os
import json
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import signal
import psutil

class FineTuneManager:
    def __init__(self):
        self.current_process = None
        self.training_logs = []
        self.training_status = "idle"  # idle, running, completed, error
        self.output_dir = "output"
        self.log_file = None
        
        # 创建必要目录
        Path(self.output_dir).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def generate_output_path(self) -> str:
        """生成输出路径"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.output_dir}/v0-{timestamp}"
    
    def build_sft_command(self, config: Dict[str, Any]) -> List[str]:
        """构建swift sft命令"""
        cmd = ["swift", "sft"]
        
        # 基础参数
        cmd.extend(["--model", config["model"]])
        cmd.extend(["--train_type", config["train_type"]])
        cmd.extend(["--torch_dtype", config["torch_dtype"]])
        
        # 数据集参数
        if config["datasets"]:
            for dataset in config["datasets"]:
                cmd.extend(["--dataset", dataset])
        
        # 训练参数
        cmd.extend(["--num_train_epochs", str(config["num_train_epochs"])])
        cmd.extend(["--per_device_train_batch_size", str(config["per_device_train_batch_size"])])
        cmd.extend(["--per_device_eval_batch_size", str(config["per_device_eval_batch_size"])])
        cmd.extend(["--learning_rate", str(config["learning_rate"])])
        cmd.extend(["--gradient_accumulation_steps", str(config["gradient_accumulation_steps"])])
        
        # LoRA参数
        if config["train_type"] == "lora":
            cmd.extend(["--lora_rank", str(config["lora_rank"])])
            cmd.extend(["--lora_alpha", str(config["lora_alpha"])])
            cmd.extend(["--target_modules", config["target_modules"]])
        
        # 其他参数
        cmd.extend(["--eval_steps", str(config["eval_steps"])])
        cmd.extend(["--save_steps", str(config["save_steps"])])
        cmd.extend(["--save_total_limit", str(config["save_total_limit"])])
        cmd.extend(["--logging_steps", str(config["logging_steps"])])
        cmd.extend(["--max_length", str(config["max_length"])])
        cmd.extend(["--warmup_ratio", str(config["warmup_ratio"])])
        cmd.extend(["--dataloader_num_workers", str(config["dataloader_num_workers"])])
        
        # 输出目录
        output_path = self.generate_output_path()
        cmd.extend(["--output_dir", output_path])
        
        # 模型信息
        if config.get("system"):
            cmd.extend(["--system", config["system"]])
        if config.get("model_author"):
            cmd.extend(["--model_author", config["model_author"]])
        if config.get("model_name"):
            cmd.extend(["--model_name", config["model_name"]])
            
        return cmd
    
    def start_training(self, config: Dict[str, Any]) -> str:
        """启动训练"""
        if self.training_status == "running":
            return "训练已在进行中"
        
        try:
            cmd = self.build_sft_command(config)
            
            # 设置环境变量
            env = os.environ.copy()
            if config.get("cuda_visible_devices"):
                env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
            
            # 创建日志文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path("logs") / f"training_{timestamp}.log"
            
            # 启动进程
            self.current_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.training_status = "running"
            self.training_logs = []
            
            # 启动日志监控线程
            threading.Thread(target=self._monitor_training, daemon=True).start()
            
            return f"训练已启动\n命令: {' '.join(cmd)}"
            
        except Exception as e:
            self.training_status = "error"
            return f"启动训练失败: {str(e)}"
    
    def _monitor_training(self):
        """监控训练过程"""
        try:
            with open(self.log_file, 'w') as f:
                for line in iter(self.current_process.stdout.readline, ''):
                    if line:
                        self.training_logs.append(line.strip())
                        f.write(line)
                        f.flush()
                        
                        # 保持最近1000行日志
                        if len(self.training_logs) > 1000:
                            self.training_logs = self.training_logs[-1000:]
            
            self.current_process.wait()
            
            if self.current_process.returncode == 0:
                self.training_status = "completed"
            else:
                self.training_status = "error"
                
        except Exception as e:
            self.training_status = "error"
            self.training_logs.append(f"监控错误: {str(e)}")
    
    def stop_training(self) -> str:
        """停止训练"""
        if self.current_process and self.training_status == "running":
            try:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                
                self.training_status = "idle"
                return "训练已停止"
            except Exception as e:
                return f"停止训练失败: {str(e)}"
        else:
            return "没有正在运行的训练"
    
    def get_training_logs(self) -> str:
        """获取训练日志"""
        return "\n".join(self.training_logs[-50:])
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            "status": self.training_status,
            "log_count": len(self.training_logs),
            "process_alive": self.current_process is not None and self.current_process.poll() is None
        }
    
    def get_available_checkpoints(self) -> List[str]:
        """获取可用的checkpoint"""
        checkpoints = []
        output_dir = Path(self.output_dir)
        
        if output_dir.exists():
            for version_dir in output_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith("v0-"):
                    for checkpoint_dir in version_dir.iterdir():
                        if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("checkpoint-"):
                            checkpoints.append(str(checkpoint_dir))
        
        return sorted(checkpoints, reverse=True)

# 全局实例
ft_manager = FineTuneManager()