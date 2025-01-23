# kubeflow_ci/scripts/train_final.py
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    """
    Usage: train_final.py <lr> <batch_size>
    """
    if len(sys.argv) < 3:
        print("Usage: train_final.py <lr> <batch_size>")
        os._exit(1)

    lr = float(sys.argv[1])
    batch_size = int(sys.argv[2])

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

    # Shuffle data
    n = len(df)
    idx = torch.randperm(n)
    X = X[idx]
    y = y[idx]

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
    train_size = len(X)
    num_batches = max(1, train_size // batch_size)

    for epoch in range(epochs):
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            xb = X[start:end]
            yb = y[start:end]

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    # Save model
    os.makedirs("/workspace/model", exist_ok=True)
    model_path = "/workspace/model/model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path} with lr={lr}, batch_size={batch_size}")

if __name__ == "__main__":
    main()