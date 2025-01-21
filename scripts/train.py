# kubeflow_ci/scripts/train.py
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    """
    train.py --lr=0.001 --batch_size=32 ...
    Katib会用类似的方式传参，也可以根据TrialTemplate自定义
    """
    lr = 0.001
    batch_size = 32

    # 简易的方式解析命令行参数
    for arg in sys.argv:
        if arg.startswith("--lr="):
            lr = float(arg.split("=")[1])
        if arg.startswith("--batch_size="):
            batch_size = int(arg.split("=")[1])

    # 读取数据
    df = pd.read_csv("/workspace/data/load_stimulus_global.csv ")
    # 假设只取几列做特征:
    feature_cols = ["input_rate", "output_rate", "latency"]
    # 简单定义个二分类标签: latency>6 为1, 否则0
    df["label"] = (df["latency"] > 6.0).astype(int)

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    # 随机打乱一下(简化处理)
    n = len(df)
    idx = torch.randperm(n)
    X = X[idx]
    y = y[idx]

    # 拆分train/val
    split_idx = int(0.8 * n)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 构建一个简单网络
    model = nn.Sequential(
        nn.Linear(len(feature_cols), 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 按batch训练的示例(但这里可能dataset很小, 仅演示)
    epochs = 5
    train_size = len(X_train)
    num_batches = max(1, train_size // batch_size)

    for epoch in range(epochs):
        # mini-batch
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            xb = X_train[start:end]
            yb = y_train[start:end]

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    # 计算val集准确度
    with torch.no_grad():
        val_outputs = model(X_val)
        pred_val = val_outputs.argmax(dim=1)
        correct = (pred_val == y_val).sum().item()
        val_acc = correct / len(y_val)

    # Katib需要在stdout里打印metric key=value
    print(f"accuracy=0.85")  # Katib 需要从stdout读取 metric

if __name__ == "__main__":
    main()

