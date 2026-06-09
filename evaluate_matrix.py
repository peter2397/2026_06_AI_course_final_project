import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report  # 加入了 classification_report
from dataset import get_data_loaders
from model import LightWeightCNN
import os


def evaluate_model():
    # 0. 檢查模型檔案是否存在
    if not os.path.exists('component_model.pth'):
        print("❌ 找不到 'component_model.pth'！請先執行 train.py 完成模型訓練。")
        return

    # 1. 喚醒 GPU 與模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動模型評估系統！目前使用的運算裝置: {device}")

    model = LightWeightCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load('component_model.pth', map_location=device))
    model.eval()  # 切換到考試模式，關閉 Dropout 等機制

    # 2. 拿取驗證集資料 (期中考卷)
    print("📦 正在載入資料集進行測驗...")
    # 只需要 val_loader (驗證集) 和 classes (類別名稱) 來做評估
    _, val_loader, classes = get_data_loaders(batch_size=16)

    y_true = []
    y_pred = []

    # 3. 開始作答 (推論階段)
    print("🧠 模型正在進行推論，請稍候...")
    with torch.no_grad():  # 評估時不需要計算梯度，節省記憶體與加速
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 將結果轉回 CPU 並存入陣列中
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # ==========================================
    # 📊 產出報告一：進階分類能力分析報表 (終端機顯示)
    # ==========================================
    print("\n" + "=" * 50)
    print("📊 終極分類能力分析報表 (Classification Report)")
    print("=" * 50)
    # 自動計算 Precision, Recall, F1-Score
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    print("=" * 50)
    print("💡 提示：您可以將上方報表複製或截圖放入 PPT，F1-Score 是學術界最看重的指標！\n")

    # ==========================================
    # 🎨 產出報告二：精美彩色混淆矩陣圖 (存成圖片)
    # ==========================================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    # 設定字體以支援中文顯示 (Windows 預設使用微軟正黑體)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 繪製熱力圖
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})  # 數字字體大小

    plt.title('電子元件分類 - 混淆矩陣 (Confusion Matrix)', fontsize=18, pad=20)
    plt.ylabel('真實標籤 (True Label)', fontsize=14)
    plt.xlabel('模型預測 (Predicted Label)', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 自動調整排版並存檔
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("🎉 混淆矩陣圖已成功儲存為專案目錄下的 'confusion_matrix.png'！")

    # 顯示圖片視窗
    plt.show()


if __name__ == '__main__':
    evaluate_model()