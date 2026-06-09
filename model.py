import torch.nn as nn


class LightWeightCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(LightWeightCNN, self).__init__()

        # --- 特徵萃取層 (Feature Extraction) ---
        # 就像 AI 的眼睛，負責從 128x128 的圖片中找出邊緣、形狀等特徵
        self.features = nn.Sequential(
            # 第一層：萃取淺層特徵 (如直線、邊緣)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 影像縮小成 64x64

            # 第二層：萃取中層特徵 (如圓角、紋理)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 影像縮小成 32x32

            # 第三層：萃取深層特徵 (如複雜的幾何組合)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 影像縮小成 16x16
        )

        # --- 分類層 (Classifier) ---
        # 就像 AI 的大腦，根據眼睛看到的特徵來決定這到底是什麼元件
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 把立體的特徵圖攤平成一維陣列
            nn.Linear(64 * 16 * 16, 128),  # 64層通道 * 長16 * 寬16
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 【關鍵】隨機關閉 50% 的神經元，強迫 AI 不要死背照片！
            nn.Linear(128, num_classes)  # 最後輸出 5 個類別的機率得分
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x