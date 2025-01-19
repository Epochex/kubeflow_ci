# Dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝脚本到容器
COPY scripts/ ./scripts/

# 我们默认的ENTRYPOINT是bash, 具体Task里会自定义command
ENTRYPOINT ["/bin/bash"]
