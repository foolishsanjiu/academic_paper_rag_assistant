"""Minimal PyTorch example for Day 08."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyRegressionDataset(Dataset):
    """A tiny dataset following approximately y = 2x + 1."""

    def __init__(self) -> None:
        self.x = torch.tensor(
            [[1.0], [2.0], [3.0], [4.0], [5.0]]
        )

        self.y = torch.tensor(
            [[3.0], [5.0], [7.0], [9.0], [11.0]]
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")

    dataset = TinyRegressionDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    model = nn.Linear(
        in_features=1,
        out_features=1,
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    # -------------------------
    # Training
    # -------------------------

    model.train()

    for epoch in range(100):
        total_loss = 0.0

        for x_batch, y_batch in dataloader:

            # 1. 前向传播
            prediction = model(x_batch)

            # 2. 计算损失
            loss = loss_function(
                prediction,
                y_batch,
            )

            # 3. 清空上一轮梯度
            optimizer.zero_grad()

            # 4. 反向传播
            loss.backward()

            # 5. 更新模型参数
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d} "
                f"| Loss: {total_loss:.6f}"
            )

    # -------------------------
    # Inference
    # -------------------------

    model.eval()

    test_x = torch.tensor([[6.0]])

    with torch.no_grad():
        prediction = model(test_x)

    print()
    print(f"输入 x = {test_x.item()}")
    print(
        f"模型预测 y = "
        f"{prediction.item():.4f}"
    )


if __name__ == "__main__":
    main()