import gradio as gr
from core.data_manager import data_manager
from ui.html_templates import nav_html

def get_data_upload_block():
    """数据上传界面"""
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
        
        gr.Markdown("# 📁 训练数据管理", elem_classes=["text-center"])
        
        with gr.Tab("📤 上传数据集"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 选择数据文件")
                    upload_files = gr.Files(
                        label="支持格式: .jsonl, .json, .csv, .txt",
                        file_types=[".jsonl", ".json", ".csv", ".txt"],
                        file_count="multiple"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 数据集信息")
                    dataset_name = gr.Textbox(
                        label="数据集名称",
                        placeholder="输入数据集名称",
                        max_lines=1
                    )
                    upload_btn = gr.Button(
                        "🚀 上传数据集", 
                        variant="primary",
                        size="lg"
                    )
            
            upload_result = gr.Textbox(
                label="上传结果",
                interactive=False,
                lines=3
            )
        
        with gr.Tab("✏️ 创建数据集"):
            gr.Markdown("### 批量输入训练数据")
            gr.Markdown("💡 **提示**: 每行输入一条数据，三个文本框的行数必须相同")
            
            with gr.Row():
                with gr.Column():
                    instructions = gr.Textbox(
                        label="指令列表 (每行一个)",
                        lines=8,
                        placeholder="请翻译以下文本\n请总结以下内容\n请回答以下问题"
                    )
                    
                with gr.Column():
                    inputs = gr.Textbox(
                        label="输入列表 (每行一个)",
                        lines=8,
                        placeholder="Hello world\n这是一篇关于AI的文章...\n什么是机器学习？"
                    )
                    
                with gr.Column():
                    outputs = gr.Textbox(
                        label="输出列表 (每行一个)",
                        lines=8,
                        placeholder="你好世界\n文章总结了AI的发展历程...\n机器学习是人工智能的一个分支..."
                    )
            
            with gr.Row():
                custom_dataset_name = gr.Textbox(
                    label="数据集名称",
                    placeholder="输入自定义数据集名称",
                    scale=3
                )
                create_btn = gr.Button(
                    "✨ 创建数据集",
                    variant="primary",
                    scale=1
                )
            
            create_result = gr.Textbox(
                label="创建结果",
                interactive=False,
                lines=3
            )
        
        with gr.Tab("📋 管理数据集"):
            with gr.Row():
                with gr.Column(scale=1):
                    dataset_dropdown = gr.Dropdown(
                        choices=data_manager.get_available_datasets(),
                        label="选择数据集",
                        interactive=True
                    )
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                        preview_btn = gr.Button("👀 预览数据", size="sm")
                        delete_btn = gr.Button("🗑️ 删除数据集", variant="stop", size="sm")
                    
                    manage_result = gr.Textbox(
                        label="操作结果",
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column(scale=2):
                    dataset_preview = gr.Textbox(
                        label="数据集预览",
                        lines=15,
                        interactive=False
                    )
        
        # 事件绑定
        upload_btn.click(
            fn=data_manager.upload_dataset,
            inputs=[upload_files, dataset_name],
            outputs=[upload_result]
        )
        
        create_btn.click(
            fn=data_manager.create_custom_dataset,
            inputs=[instructions, inputs, outputs, custom_dataset_name],
            outputs=[create_result]
        )
        
        refresh_btn.click(
            fn=data_manager.get_available_datasets,
            outputs=[dataset_dropdown]
        )
        
        preview_btn.click(
            fn=data_manager.preview_dataset,
            inputs=[dataset_dropdown],
            outputs=[dataset_preview]
        )
        
        delete_btn.click(
            fn=data_manager.delete_dataset,
            inputs=[dataset_dropdown],
            outputs=[manage_result]
        )
    
    return demo