kubeflow_ci
├── data
│   └── dataset.csv              # 你的原始数据集
├── katib
│   └── katib-experiment.yaml    # Katib Experiment配置
├── scripts
│   ├── train.py                 # 普通训练脚本
│   └── train_final.py           # 用最佳超参数做最终训练的脚本
├── tekton
│   ├── tasks
│   │   ├── build-image-task.yaml         # 用于构建并推送Docker镜像
│   │   ├── run-katib-task.yaml           # 启动并等待Katib完成的Task
│   │   ├── train-with-best-params.yaml   # 根据Katib最佳超参进行最终训练
│   └── pipeline.yaml            # 把多个Task拼成Pipeline
│   └── triggers.yaml            # Tekton Trigger, 监听Git事件自动触发PipelineRun
├── Dockerfile                   # 用于构建训练镜像
├── requirements.txt             # Python依赖
└── README.md


# 进入 kubeflow_ci 目录
cd /data/mlops/kubeflow_ci

# 创建文件夹和文件
# 1. 创建 data 文件夹并创建 dataset.csv 文件
mkdir -p data
touch data/dataset.csv

# 2. 创建 katib 文件夹并创建 katib-experiment.yaml 文件
mkdir -p katib
touch katib/katib-experiment.yaml

# 3. 创建 scripts 文件夹并创建 train.py 和 train_final.py 文件
mkdir -p scripts
touch scripts/train.py
touch scripts/train_final.py

# 4. 创建 tekton/tasks 文件夹并创建对应的 YAML 文件
mkdir -p tekton/tasks
touch tekton/tasks/build-image-task.yaml
touch tekton/tasks/run-katib-task.yaml
touch tekton/tasks/train-with-best-params.yaml

# 5. 创建 pipeline.yaml 和 triggers.yaml 文件
touch tekton/pipeline.yaml
touch tekton/triggers.yaml

# 6. 创建 Dockerfile 和 requirements.txt 文件
touch Dockerfile
touch requirements.txt

# 7. 创建 README.md 文件
touch README.md

# 验证目录结构
tree .

连接远程主机，连接时连不上删除旧密钥
ssh-keygen -R 45.149.207.13



# 安装 Tekton Pipelines 和triggers(自动触发功能)           最新发布版
kubectl apply -f https://storage.googleapis.com/tekton-releases/pipeline/latest/release.yaml
kubectl apply -f https://storage.googleapis.com/tekton-releases/triggers/latest/release.yaml

# 可选）安装 Tekton Dashboard：
kubectl apply --filename https://storage.googleapis.com/tekton-releases/dashboard/latest/release-full.yaml
kubectl get pods --namespace tekton-pipelines --watch

# 确认tekton api版本
kubectl api-versions | grep tekton.dev

# 准备CI
cd /data/mlops/kubeflow_ci

# 如果还没git init过
git init

git remote add origin https://github.com/<your-github-account>/kubeflow_ci.git
# 或使用 SSH URL: git@github.com:<your-github-account>/kubeflow_ci.git

git add .
git commit -m "Initial commit"
git push -u origin master

Developer settings → Personal access tokens → Tokens (classic)。
use token

私有/公共仓库都可以。若是私有，需要在后续 Tekton Triggers 中配置 GitHub Token/Secret 用于 Webhook 验证，这里先简化不做配置。

# token 记忆配置
git config --global credential.helper store
之后再次推送时，输入用户名和令牌，Git 会自动保存到 ~/.git-credentials 文件中。

# 要让 Tekton（Kaniko）能推到 Docker Hub，需要在集群里创建一个docker-registry类型的 Secret，然后给 Pipeline/Task 绑定
查看当前serviceaccount
kubectl get serviceaccount -n default
如果没有pipeline的serviceaccount话需要创建一个
kubectl create serviceaccount pipeline -n default


# 1) 在 K8s 集群里创建 Secret
kubectl create secret docker-registry docker-cred \
  --docker-username=hirschazer \
  --docker-password=linjianke830dock \
  --docker-email=crayozakka@gmail.com \
  --docker-server=https://index.docker.io/v1/ \
  -n default

# 2) 绑定到默认 ServiceAccount (pipeline)
kubectl patch serviceaccount pipeline \
  -p '{"imagePullSecrets": [{"name": "docker-cred"}]}' \
  -n default

这样Kaniko 在推镜像时，就能拿到 Docker Hub 的认证。






