import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict
import random

from model import LightWeightCNN


def main_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動交叉驗證系統！運算裝置: {device}")

    # 1. 載入並平衡資料 (擷取自 dataset.py 的邏輯)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = ImageFolder(root='./dataset', transform=transform)

    class_indices = defaultdict(list)
    for idx, (_, class_id) in enumerate(full_dataset.samples):
        class_indices[class_id].append(idx)

    selected_indices = []
    for class_id, indices in class_indices.items():
        num_to_sample = min(200, len(indices))
        selected_indices.extend(random.sample(indices, num_to_sample))
    balanced_dataset = Subset(full_dataset, selected_indices)

    # 2. 設定 K-Fold (切 5 份)
    k_folds = 5
    num_epochs = 20  # 為了節省時間，交叉驗證時 Epoch 設為 20 即可
    kfold = KFold(n_splits=k_folds, shuffle=True)

    results = {}
    print(f"📊 準備進行 {k_folds}-Fold 交叉驗證...\n" + "-" * 40)

    # 3. 開始 K-Fold 迴圈
    for fold, (train_ids, val_ids) in enumerate(kfold.split(balanced_dataset)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        # 定義這個 Fold 的發牌機 (DataLoaders)
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(balanced_dataset, batch_size=16, sampler=train_subsampler)
        val_loader = DataLoader(balanced_dataset, batch_size=16, sampler=val_subsampler)

        # ⚠️ 【關鍵】每個 Fold 都必須產生一顆「全新」的腦，不能繼承上一次的記憶
        model = LightWeightCNN(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 開始這個 Fold 的訓練馬拉松
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 這個 Fold 訓練結束，進行期中考
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f'✅ Fold {fold + 1} 驗證準確率: {accuracy:.2f} %')
        results[fold] = accuracy

    # 4. 印出最終總結報告
    print("\n" + "=" * 40)
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
    print("=" * 40)
    sum_acc = 0.0
    for key, value in results.items():
        print(f'Fold {key + 1}: {value:.2f} %')
        sum_acc += value
    print(f'⭐ 平均準確率 (Average Accuracy): {sum_acc / k_folds:.2f} %')
    print("=" * 40)


if __name__ == '__main__':
    main_cv()