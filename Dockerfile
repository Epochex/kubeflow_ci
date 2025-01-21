# /data/mlops/kubeflow_ci/Dockerfile
# FROM python:3.11

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# # 将脚本拷贝到镜像
# COPY scripts/ ./scripts/
# COPY data/load_stimulus_global.csv /workspace/data/
# # 使用 CMD 保持容器运行
# CMD ["tail", "-f", "/dev/null"]

# 基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制项目的所有内容到镜像中
COPY . /app

# 安装依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 设置默认命令以保持容器运行
CMD ["tail", "-f", "/dev/null"]

# 修改后重新构建指令
# docker build -t docker.io/hirschazer/kubeflow_ci:latest /data/mlops/kubeflow_ci
# 然后再重新push
# docker push docker.io/hirschazer/kubeflow_ci:latest
