"""手写数字识别：训练、验证选模、测试与错例检查。"""

import argparse
import json
from pathlib import Path

import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HIDDEN_DIM = 64
EPOCHS = 50
SEED = 42


def run(hidden_dim=HIDDEN_DIM, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )
    X_train, X_val, X_test = [
        torch.tensor(values, dtype=torch.float32)
        for values in (X_train, X_val, X_test)
    ]
    y_train, y_val, y_test = [
        torch.tensor(values, dtype=torch.long)
        for values in (y_train, y_val, y_test)
    ]
    print(f"训练集 {X_train.shape}，验证集 {X_val.shape}，测试集 {X_test.shape}")
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    model = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_accuracy = -1.0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        model.eval()
        with torch.no_grad():
            val_accuracy = (model(X_val).argmax(1) == y_val).float().mean().item()
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        average_loss = total_loss / len(X_train)
        history.append({"epoch": epoch + 1, "train_loss": average_loss, "val_accuracy": val_accuracy})
        print(f"第 {epoch + 1:02d} 轮 | 训练损失：{average_loss:.3f} | 验证准确率：{val_accuracy:.1%}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).argmax(1)
        test_accuracy = (predictions == y_test).float().mean().item()
    wrong = torch.where(predictions != y_test)[0]
    print(f"\n选中第 {best_epoch} 轮的模型")
    print(f"最佳验证准确率：{best_val_accuracy:.1%}")
    print(f"最终测试准确率：{test_accuracy:.1%}")
    print(f"测试集共 {len(y_test)} 张，认错 {len(wrong)} 张")
    for index in wrong[:3]:
        i = index.item()
        print(f"\n真实数字：{y_test[i].item()}，模型预测：{predictions[i].item()}")
        for row in X_test[i].reshape(8, 8):
            print("".join("██" if pixel > 0.5 else "  " for pixel in row))

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    name = f"hidden{hidden_dim}_epochs{epochs}_seed{seed}"
    record = {
        "hidden_dim": hidden_dim, "epochs": epochs, "seed": seed,
        "learning_rate": 0.001, "batch_size": 64,
        "train_size": len(y_train), "val_size": len(y_val), "test_size": len(y_test),
        "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
        "best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy, "history": history,
        "wrong_examples": [{"test_index": i.item(), "label": y_test[i].item(),
                            "prediction": predictions[i].item()} for i in wrong],
    }
    (results / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"hidden_dim": hidden_dim, "state_dict": best_state}, results / f"{name}.pth")
    print(f"\n实验记录及模型保存至：{results}")
    return record


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-dim", type=positive_int, default=HIDDEN_DIM)
    parser.add_argument("--epochs", type=positive_int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.hidden_dim, args.epochs, args.seed)
