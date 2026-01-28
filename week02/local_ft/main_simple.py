from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr
from ui.data_upload import get_data_upload_block
from ui.fine_tune import get_fine_tune_block
from ui.model_merge import get_model_merge_block
from ui.quantization import get_quantization_block
from ui.html_templates import main_html

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_main():
    return HTMLResponse(content=main_html)

# 挂载核心功能模块
app = gr.mount_gradio_app(app, get_data_upload_block(), path="/data_upload")
app = gr.mount_gradio_app(app, get_fine_tune_block(), path="/fine_tune")
app = gr.mount_gradio_app(app, get_model_merge_block(), path="/model_merge")
app = gr.mount_gradio_app(app, get_quantization_block(), path="/quantization")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7866)