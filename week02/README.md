# 进入代码目录
```
cd week02
```

# 如何安装依赖
本次由于有torch依赖，而torch根据不同的cpu架构又要求不同的版本，所以不提供requirements.txt或uv.lock文件，同学们请根据下面的命令生成自己的requirement.txt或uv.lock

## 使用Conda的看这里
```
conda create --name ai-engineer-week02 python=3.11
conda activate ai-engineer-week02
pip install uv
uv pip compile pyproject.toml --output-file requirements.txt
uv pip install -r requirements.txt
```

## 不想使用Conda，想直接使用uv的看这里
```
pip install uv
uv sync
source .venv/bin/activate 
```

## uv sync时, Intel芯片Mac可能会遇到的问题
```
$ uv sync         
Resolved 146 packages in 113ms
error: Distribution `torch==2.7.1 @ registry+https://pypi.tuna.tsinghua.edu.cn/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform

hint: You're on macOS (`macosx_14_0_x86_64`), but `torch` (v2.7.1) only has wheels for the following platforms: `manylinux_2_28_aarch64`, `manylinux_2_28_x86_64`, `macosx_11_0_arm64`, `win_amd64`; consider adding your platform to `tool.uv.required-environments` to ensure uv resolves to a version with compatible wheels
(base) 
```
出现该问题时，可以尝试将pyproject.toml里的torch版本修改为==2.2.2，再尝试uv sync

# 运行项目
```
cd week02
python -m local_ft.server
```

# 从浏览器访问
```
http://localhost:7866
```