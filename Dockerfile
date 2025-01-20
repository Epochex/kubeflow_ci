# /data/mlops/kubeflow_ci/Dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 将脚本拷贝到镜像
COPY scripts/ ./scripts/

# 使用 CMD 保持容器运行
CMD ["tail", "-f", "/dev/null"]
