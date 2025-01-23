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
    Katib will pass these CLI arguments in each Trial.
    """
    lr = 0.001
    batch_size = 32

    # Parse CLI arguments
    for arg in sys.argv:
        if arg.startswith("--lr="):
            lr = float(arg.split("=")[1])
        if arg.startswith("--batch_size="):
            batch_size = int(arg.split("=")[1])

    csv_path = "/workspace/data/load_stimulus_global.csv"

    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at {csv_path}")
        os._exit(1)

    # Load data
    df = pd.read_csv(csv_path)
    feature_cols = ["input_rate", "output_rate", "latency"]
    df["label"] = (df["latency"] > 6.0).astype(int)

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    # Shuffle and split data
    n = len(df)
    idx = torch.randperm(n)
    X = X[idx]
    y = y[idx]

    split_idx = int(0.8 * n)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Build model
    model = nn.Sequential(
        nn.Linear(len(feature_cols), 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    epochs = 5
    train_size = len(X_train)
    num_batches = max(1, train_size // batch_size)

    for epoch in range(epochs):
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

    # Validate model
    with torch.no_grad():
        val_outputs = model(X_val)
        pred_val = val_outputs.argmax(dim=1)
        correct = (pred_val == y_val).sum().item()
        val_acc = correct / len(y_val)

    # Log accuracy to file (Katib's metrics collector will read this)
    log_path = "/var/log/katib/metrics.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write(f"accuracy={val_acc}\n")
        print(f"Logged accuracy={val_acc} to {log_path}")

if __name__ == "__main__":
    main()
