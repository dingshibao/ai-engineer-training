# 🤖 AI模型微调平台 - 简化版

## 概述

专注于核心微调功能的简洁高效平台，移除RAG相关组件，提供清晰直观的微调工作流。

## 🎯 核心功能

### 1. 📁 数据上传
- 支持多种格式训练数据上传 (JSONL, JSON, CSV, TXT)
- 在线创建自定义数据集
- 数据预览和管理

### 2. 🚀 模型微调  
- 基于ms-swift的LoRA微调
- 实时训练监控和Loss可视化
- 灵活的参数配置

### 3. 🔗 权重合并
- LoRA权重与基础模型合并
- 支持多种精度设置
- 合并模型管理

### 4. 🗜️ 模型量化
- INT8/INT4量化支持
- 多种量化方法 (BNB, GPTQ, AWQ)
- 灵活的校准数据集配置

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements_simple.txt
```

### 2. 安装ms-swift
```bash
pip install ms-swift -U
```

### 3. 启动平台
```bash
python start_simple.py
```

### 4. 访问界面
- 主页: http://127.0.0.1:7866
- 数据上传: http://127.0.0.1:7866/data_upload
- 模型微调: http://127.0.0.1:7866/fine_tune
- 权重合并: http://127.0.0.1:7866/model_merge
- 模型量化: http://127.0.0.1:7866/quantization

## 📋 使用流程

### 步骤1: 准备数据
1. 访问"数据上传"页面
2. 上传JSONL格式训练数据或在线创建数据集
3. 预览和验证数据格式

### 步骤2: 模型微调
1. 访问"模型微调"页面
2. 配置模型和训练参数
3. 开始训练并实时监控

### 步骤3: 权重合并
1. 访问"权重合并"页面
2. 选择训练完成的checkpoint
3. 合并LoRA权重到基础模型

### 步骤4: 模型量化
1. 访问"模型量化"页面
2. 选择合并后的模型
3. 配置量化参数并执行

## 🔧 技术架构

```
├── main_simple.py          # 主程序入口
├── core/                   # 核心功能模块
│   ├── fine_tune_manager.py   # 微调管理
│   └── data_manager.py        # 数据管理
├── ui/                     # 界面模块
│   ├── html_templates.py      # HTML模板
│   ├── data_upload.py         # 数据上传界面
│   ├── fine_tune.py           # 微调界面
│   ├── model_merge.py         # 权重合并界面
│   └── quantization.py        # 量化界面
└── requirements_simple.txt # 简化依赖列表
```

## 📊 数据格式

### JSONL格式示例
```json
{"instruction": "请翻译以下文本", "input": "Hello world", "output": "你好世界"}
{"instruction": "请总结以下内容", "input": "这是一篇关于AI的文章...", "output": "文章总结了AI的发展历程..."}
```

## ⚙️ 配置说明

### 微调参数
- **学习率**: 建议1e-4到1e-5
- **LoRA Rank**: 建议8-16
- **批次大小**: 根据显存调整
- **训练轮数**: 通常1-3轮

### 量化选择
- **INT8**: 平衡精度和性能
- **INT4**: 最大压缩比
- **BNB**: 通用兼容性好
- **GPTQ/AWQ**: 高精度量化

## 🔍 故障排除

### 常见问题
1. **swift命令未找到**: `pip install ms-swift -U`
2. **CUDA内存不足**: 减小batch_size或使用梯度累积
3. **量化失败**: 检查模型路径和量化参数
4. **合并失败**: 确认checkpoint路径正确

### 日志查看
- 训练日志: `logs/training_*.log`
- 界面日志: 浏览器控制台

## 📈 性能优化

### 训练优化
- 使用合适的batch_size和gradient_accumulation_steps
- 选择适当的LoRA参数
- 监控GPU内存使用

### 推理优化
- 量化模型减少内存占用
- 使用vLLM等推理加速框架

## 🆕 版本特性

### v2.0 简化版特性
- ✅ 移除RAG相关组件
- ✅ 专注核心微调功能
- ✅ 简洁优雅的界面设计
- ✅ 模块化架构设计
- ✅ 完整的微调工作流
- ✅ 灵活的量化配置

---

**让AI模型微调更简单高效！** 🚀