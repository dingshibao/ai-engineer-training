#!/usr/bin/env python3
"""
简化版AI模型微调平台启动脚本
专注于核心微调功能：数据上传、模型微调、权重合并、模型量化
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """检查核心依赖"""
    print("检查核心依赖...")
    try:
        import gradio
        import fastapi
        import matplotlib
        print("✅ 核心依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("💡 请运行: pip install -r requirements_simple.txt")
        return False

def check_swift():
    """检查swift是否可用"""
    print("检查swift...")
    try:
        result = subprocess.run(["swift", "sft", "-h"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ms-swift 已安装")
            return True
        else:
            print("❌ swift 命令不可用")
            return False
    except Exception as e:
        print(f"❌ ms-swift 未安装或出错: {e}")
        print("💡 请运行: pip install ms-swift -U")
        return False

def create_directories():
    """创建必要目录"""
    script_dir = Path(__file__).parent.resolve()
    directories = [
        "training_data/uploaded",
        "training_data/processed", 
        "output",
        "logs",
        "merged_models",
        "quantized_models"
    ]
    
    for directory in directories:
        (script_dir / directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {script_dir / directory}")

def start_server():
    """启动简化版服务器"""
    print("🚀 启动AI模型微调平台...")
    script_dir = Path(__file__).parent.resolve()
    try:
        subprocess.run(
            ["uvicorn", "main_simple:app", "--host", "0.0.0.0", "--port", "7866", "--reload"],
            cwd=script_dir
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 AI模型微调平台 - 简化版")
    print("=" * 60)
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 检查swift
    if not check_swift():
        print("⚠️  ms-swift未安装，部分功能可能不可用")
    
    # 创建目录
    create_directories()
    
    print("\n🎯 核心功能:")
    print("• 数据上传: http://127.0.0.1:7866/data_upload")
    print("• 模型微调: http://127.0.0.1:7866/fine_tune") 
    print("• 权重合并: http://127.0.0.1:7866/model_merge")
    print("• 模型量化: http://127.0.0.1:7866/quantization")
    
    print("\n📋 使用流程:")
    print("1️⃣  上传训练数据 → 数据上传页面")
    print("2️⃣  配置并开始微调 → 模型微调页面")
    print("3️⃣  合并LoRA权重 → 权重合并页面")
    print("4️⃣  量化模型 → 模型量化页面")
    
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    # 启动服务器
    start_server()

if __name__ == "__main__":
    main()
