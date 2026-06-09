import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from model import LightWeightCNN


def train_and_plot():
    # 1. 喚醒 GPU 與載入資料
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 系統啟動！目前使用的運算裝置: {device}")

    train_loader, val_loader, classes = get_data_loaders(batch_size=16)

    # 2. 初始化模型、損失函數與優化器
    model = LightWeightCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25

    # 🌟 準備四個空陣列，用來記錄畫圖需要的歷史軌跡
    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []

    print(f"🏃‍♂️ 開始訓練，總共 {num_epochs} 個 Epoch...")

    # 3. 開始訓練迴圈
    for epoch in range(num_epochs):
        # --- 訓練階段 (Train) ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train

        # --- 驗證階段 (Validation) ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100.0 * correct_val / total_val

        # 🌟 將算出的成績記錄到歷史陣列中
        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(epoch_val_loss)
        history_train_acc.append(epoch_train_acc)
        history_val_acc.append(epoch_val_acc)

        print(f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    # 4. 儲存最新權重
    torch.save(model.state_dict(), 'component_model.pth')
    print("🎉 訓練大功告成！AI 權重已經存成 'component_model.pth'")

    # ==========================================
    # 🎨 開始繪製學習曲線 (Learning Curves)
    # ==========================================
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(14, 5))
    # 支援中文顯示
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 左圖：Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, 'b-', label='訓練誤差 (Train Loss)', linewidth=2)
    plt.plot(epochs_range, history_val_loss, 'r-', label='驗證誤差 (Val Loss)', linewidth=2)
    plt.title('誤差收斂曲線 (Loss Curve)', fontsize=16)
    plt.xlabel('訓練週期 (Epoch)', fontsize=12)
    plt.ylabel('誤差值 (Cross Entropy Loss)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 右圖：Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_train_acc, 'b-', label='訓練準確率 (Train Acc)', linewidth=2)
    plt.plot(epochs_range, history_val_acc, 'r-', label='驗證準確率 (Val Acc)', linewidth=2)
    plt.title('準確率成長曲線 (Accuracy Curve)', fontsize=16)
    plt.xlabel('訓練週期 (Epoch)', fontsize=12)
    plt.ylabel('準確率 (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 排版與存檔
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("🎉 學習曲線圖已成功儲存為 'learning_curves.png'！")

    # 顯示圖片
    plt.show()


if __name__ == '__main__':
    train_and_plot()