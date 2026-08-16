"""Minimal scaled dot-product attention example."""

import math

import torch


def main() -> None:
    # 假设有 3 个 token，每个 token 的特征维度为 2
    query = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    key = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    value = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ]
    )

    d_k = query.shape[-1]

    # 1. Q 与 K 做点积
    scores = query @ key.T

    # 2. 缩放
    scaled_scores = scores / math.sqrt(d_k)

    # 3. Softmax 得到注意力权重
    attention_weights = torch.softmax(
        scaled_scores,
        dim=-1,
    )

    # 4. 对 Value 加权求和
    output = attention_weights @ value

    print("QK^T:")
    print(scores)

    print("\nScaled scores:")
    print(scaled_scores)

    print("\nAttention weights:")
    print(attention_weights)

    print("\n每一行权重之和:")
    print(attention_weights.sum(dim=-1))

    print("\nAttention output:")
    print(output)


if __name__ == "__main__":
    main()