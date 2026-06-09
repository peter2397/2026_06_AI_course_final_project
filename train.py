import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import LightWeightCNN


def main():
    # 1. 喚醒你的 RTX 3050 Ti 顯示卡！
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 系統啟動！目前使用的運算裝置: {device}")
    if device.type == 'cpu':
        print("⚠️ 警告：沒有抓到 GPU，訓練會非常緩慢，請檢查 PyTorch 安裝版本！")

    # 2. 載入我們剛剛測試成功的資料集
    # batch_size=16 確保 4GB VRAM 不會當機
    train_loader, val_loader, classes = get_data_loaders('./dataset', batch_size=16)
    num_classes = len(classes)

    # 3. 初始化 AI 大腦、評分標準(Loss)與學習優化器(Optimizer)
    model = LightWeightCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 開始訓練馬拉松 (設定跑 25 圈 Epoch)
    num_epochs = 25
    print(f"🏃‍♂️ 開始訓練，總共 {num_epochs} 個 Epoch...")

    for epoch in range(num_epochs):
        model.train()  # 切換到「學習模式」
        running_loss = 0.0
        correct = 0
        total = 0

        # --- 【訓練階段】 ---
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 將圖片搬進顯示卡記憶體

            optimizer.zero_grad()  # 清空大腦上次的錯誤記憶
            outputs = model(images)  # 看圖片猜答案
            loss = criterion(outputs, labels)  # 計算猜錯了多少 (Loss)
            loss.backward()  # 反向傳播 (檢討為什麼會錯)
            optimizer.step()  # 更新神經網路的權重 (變得更聰明)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # --- 【驗證階段】 (期中考，不准看答案) ---
        model.eval()  # 切換到「考試模式」
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 考試時不更新權重，節省效能
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_acc = 100 * val_correct / val_total

        # 印出每一圈的成績單
        print(f"Epoch [{epoch + 1:02d}/{num_epochs}] "
              f"| Loss(誤差): {running_loss / len(train_loader):.4f} "
              f"| Train Acc(平時考): {train_acc:.2f}% "
              f"| Val Acc(期中考): {val_acc:.2f}%")

    # 5. 訓練完成，把變聰明的大腦存檔！
    torch.save(model.state_dict(), 'component_model.pth')
    print("🎉 訓練大功告成！AI 權重已經存成 'component_model.pth'")


if __name__ == '__main__':
    main()