import matplotlib.pyplot as plt
import numpy as np


def plot_classification_report():
    # 1. 準備數據 (將你的數據填入)
    classes = ['Capacitor', 'IC', 'LED', 'Resistor', 'Transistor']
    precision = [0.73, 0.95, 0.82, 1.00, 0.94]
    recall = [0.86, 0.98, 0.68, 0.98, 0.94]
    f1_score = [0.79, 0.96, 0.74, 0.99, 0.94]

    # 設定長條圖的位置與寬度
    x = np.arange(len(classes))
    width = 0.25

    # 2. 開始繪圖
    fig, ax = plt.subplots(figsize=(12, 6))

    # 支援中文顯示 (Windows 預設微軟正黑體)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 繪製三組長條圖 (可以自訂顏色)
    rects1 = ax.bar(x - width, precision, width, label='精確率 (Precision)', color='#4c72b0')
    rects2 = ax.bar(x, recall, width, label='召回率 (Recall)', color='#dd8452')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-分數 (F1-Score)', color='#55a868')

    # 3. 加上標籤、標題與自訂 X 軸
    ax.set_ylabel('分數 (0.0 - 1.0)', fontsize=14)
    ax.set_title('模型分類效能指標總覽 (Classification Metrics)', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.1)  # 讓最上方留一點空間放數字
    ax.legend(fontsize=12, loc='lower right')

    # 4. 在每一根長條圖上方顯示精確數字
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直平移 3 個像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # 5. 排版並存檔
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.savefig('classification_metrics.png', dpi=300)
    print("🎉 效能長條圖已成功儲存為 'classification_metrics.png'！")

    plt.show()


if __name__ == '__main__':
    plot_classification_report()