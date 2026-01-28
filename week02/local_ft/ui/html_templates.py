nav_html = """
<div class="nav-container">
    <div class="nav-bar">
        <a href="/data_upload" class="nav-button">
            <span class="nav-icon">📁</span>
            <span class="nav-text">数据管理</span>
        </a>
        <a href="/fine_tune" class="nav-button">
            <span class="nav-icon">🚀</span>
            <span class="nav-text">模型微调</span>
        </a>
        <a href="/model_merge" class="nav-button">
            <span class="nav-icon">🔗</span>
            <span class="nav-text">权重合并</span>
        </a>
        <a href="/quantization" class="nav-button">
            <span class="nav-icon">🗜️</span>
            <span class="nav-text">模型量化</span>
        </a>
    </div>
</div>
<style>
.nav-container {
    display: flex;
    justify-content: center;
    width: 100%;
    padding: 10px 0;
}
.nav-bar {
    display: flex;
    justify-content: space-around;
    align-items: center;
    background-color: rgba(26, 26, 26, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 8px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
    width: 80%;
    max-width: 800px;
}
.nav-button {
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    color: #f0f0f0;
    padding: 10px 15px;
    border-radius: 10px;
    transition: background-color 0.3s ease, color 0.3s ease;
    font-size: 16px;
    font-weight: 500;
}
.nav-button:hover {
    background-color: #007bff;
    color: white;
}
.nav-icon {
    margin-right: 8px;
    font-size: 20px;
}
.nav-text {
    white-space: nowrap;
}
</style>
"""

main_html = """
<div class="main-container">
    <div class="hero-section">
        <h1 class="title">🤖 AI 模型微调平台</h1>
        <p class="subtitle">一个集成了数据管理、模型微调、权重合并和模型量化的一站式平台。</p>
    </div>

    <div class="features-grid">
        <a href="/data_upload" class="feature-card">
            <h3>📁 数据管理</h3>
            <p>上传、创建和管理您的训练数据集。</p>
        </a>
        <a href="/fine_tune" class="feature-card">
            <h3>🚀 模型微调</h3>
            <p>使用 LoRA 或全量微调来训练您的模型。</p>
        </a>
        <a href="/model_merge" class="feature-card">
            <h3>🔗 权重合并</h3>
            <p>将 LoRA 权重合并到基础模型中。</p>
        </a>
        <a href="/quantization" class="feature-card">
            <h3>🗜️ 模型量化</h3>
            <p>对模型进行量化以优化性能。</p>
        </a>
    </div>

    <div class="footer">
        <p>由 Gemini 驱动的本地微调工具</p>
    </div>
</div>
<style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .main-container {
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .hero-section {
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #bbbbbb;
        max-width: 600px;
        margin: 0 auto;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    .feature-card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-decoration: none;
        color: #e0e0e0;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.7);
    }
    .feature-card h3 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .feature-card p {
        color: #aaaaaa;
    }
    .footer {
        padding: 1rem;
        color: #888;
        font-size: 0.9rem;
    }
    </style>
"""