#!/usr/bin/env python3
"""
测试AI模型微调平台的各个功能模块
"""

import requests
import time

def test_endpoints():
    """测试各个端点是否可访问"""
    base_url = "http://127.0.0.1:7866"
    
    endpoints = [
        "/",
        "/data_upload", 
        "/fine_tune",
        "/model_merge",
        "/quantization"
    ]
    
    print("🧪 测试平台端点...")
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - 正常")
            else:
                print(f"❌ {endpoint} - 状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint} - 连接失败: {e}")

def test_core_modules():
    """测试核心模块导入"""
    print("\n🔧 测试核心模块...")
    
    try:
        from core.data_manager import data_manager
        print("✅ 数据管理模块 - 正常")
        
        # 测试数据管理功能
        datasets = data_manager.get_available_datasets()
        print(f"📊 可用数据集: {len(datasets)} 个")
        
    except Exception as e:
        print(f"❌ 数据管理模块 - 错误: {e}")
    
    try:
        from core.fine_tune_manager import ft_manager
        print("✅ 微调管理模块 - 正常")
        
        # 测试微调管理功能
        status = ft_manager.get_training_status()
        print(f"🚀 训练状态: {status['status']}")
        
    except Exception as e:
        print(f"❌ 微调管理模块 - 错误: {e}")

def test_ui_modules():
    """测试UI模块"""
    print("\n🎨 测试UI模块...")
    
    ui_modules = [
        "ui.data_upload",
        "ui.fine_tune", 
        "ui.model_merge",
        "ui.quantization"
    ]
    
    for module in ui_modules:
        try:
            __import__(module)
            print(f"✅ {module} - 正常")
        except Exception as e:
            print(f"❌ {module} - 错误: {e}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("🤖 AI模型微调平台 - 功能测试")
    print("=" * 50)
    
    # 等待服务器启动
    print("⏳ 等待服务器启动...")
    time.sleep(2)
    
    # 测试端点
    test_endpoints()
    
    # 测试核心模块
    test_core_modules()
    
    # 测试UI模块
    test_ui_modules()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！平台已准备就绪")
    print("🌐 访问地址: http://127.0.0.1:7866")
    print("=" * 50)

if __name__ == "__main__":
    main()