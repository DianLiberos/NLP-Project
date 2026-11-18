import sys
import copy
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from neural_network import NeuralNetwork

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INTENT_NAMES = ["GREETING", "TRADE", "INFO", "HOSTILE", "UNKNOWN"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ [PyTorch Training] Sử dụng thiết bị: {device}")


def load_data():
    X_train_dense = sp.load_npz(PROCESSED_DIR / "X_train.npz").toarray()
    X_test_dense = sp.load_npz(PROCESSED_DIR / "X_test.npz").toarray()
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    X_train_tensor = torch.FloatTensor(X_train_dense).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_dense).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False
    )
    return train_loader, test_loader, y_test


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(y)
            all_preds.extend(pred.argmax(dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds) * 100
    macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100
    return avg_loss, acc, macro_f1, all_preds


def train_single_architecture(activation_function: str, epochs: int = 50, lr: float = 0.001, patience: int = 12):
    print("\n" + "=" * 65)
    print(f"🚀 HUẤN LUYỆN KIẾN TRÚC: PyTorch MLP ({activation_function})")
    print(f"   Hyperparameters: lr={lr}, epochs={epochs}, patience={patience}, batch_size=32")
    print("=" * 65)

    train_loader, test_loader, y_test = load_data()

    model = NeuralNetwork(activation_function=activation_function).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_loss = float("inf")
    best_f1 = 0.0
    best_epoch = 0
    best_state_dict = None
    best_preds = None
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_train_loss += loss.item() * len(y_batch)

        train_loss = running_train_loss / len(train_loader.dataset)
        test_loss, test_acc, test_f1, preds = evaluate(model, test_loader, loss_fn)

        is_best = False
        # Tiêu chí: Accuracy cao hơn hoặc (Acc bằng nhưng Loss thấp hơn)
        if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
            best_acc = test_acc
            best_loss = test_loss
            best_f1 = test_f1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_preds = preds
            is_best = True
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        marker = " 🌟 [BEST]" if is_best else ""
        if epoch % 5 == 0 or is_best or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:5.2f}% | "
                f"Macro F1: {test_f1:5.2f}%{marker}"
            )

        # Early Stopping
        if no_improve_epochs >= patience:
            print(f"⏹️ Early stopping kích hoạt tại Epoch {epoch} (không cải thiện sau {patience} epochs).")
            break

    print("\n" + "-" * 55)
    print(f"🏆 KẾT QUẢ ĐỈNH CAO NHẤT ({activation_function}):")
    print(f"   Epoch đạt đỉnh: {best_epoch}")
    print(f"   Accuracy:       {best_acc:.2f}%")
    print(f"   Macro F1:       {best_f1:.2f}%")
    print(f"   Test Loss:      {best_loss:.4f}")
    print("-" * 55)

    print(f"\n📊 Báo cáo phân loại chi tiết ({activation_function} tại Epoch {best_epoch}):")
    print(classification_report(y_test, best_preds, target_names=INTENT_NAMES, digits=4))

    # Lưu Best Model Checkpoint
    save_dir = CURRENT_DIR / "models" / activation_function
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{activation_function}.pth"
    torch.save(best_state_dict, save_path)
    print(f"💾 Đã lưu trọng số tốt nhất vào: {save_path}\n")

    return {
        "model": activation_function,
        "best_epoch": best_epoch,
        "accuracy": best_acc,
        "macro_f1": best_f1,
        "loss": best_loss
    }


def main():
    architectures = ["ReLU", "SiLU"]
    summary = []

    for arch in architectures:
        res = train_single_architecture(arch, epochs=40, lr=0.001, patience=10)
        summary.append(res)

    print("\n" + "=" * 65)
    print(f"{'Mô hình PyTorch':<20} | {'Best Epoch':<12} | {'Accuracy (%)':<14} | {'Macro F1 (%)':<12}")
    print("-" * 65)
    for r in summary:
        print(f"MLP ({r['model']:<14}) | {r['best_epoch']:<12} | {r['accuracy']:<14.2f} | {r['macro_f1']:<12.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
