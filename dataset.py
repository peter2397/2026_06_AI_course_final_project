import torch
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict


def get_data_loaders(data_dir='./dataset', batch_size=16, samples_per_class=200):
    """
    自動平衡版資料載入器：會自動從每個類別中隨機抽取指定數量的圖片 (預設 200 張)。
    """

    # 1. 定義資料前處理與擴增
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 讀取整個資料夾
    full_dataset = ImageFolder(root=data_dir, transform=train_transform)

    # 3. 【新增功能】自動抽樣平衡機制 (Auto-Balancing)
    # 將所有圖片依照類別分類
    class_indices = defaultdict(list)
    for idx, (_, class_id) in enumerate(full_dataset.samples):
        class_indices[class_id].append(idx)

    selected_indices = []
    print(f"🔍 正在執行資料自動平衡 (每個類別抽取 {samples_per_class} 張)...")

    for class_id, indices in class_indices.items():
        class_name = full_dataset.classes[class_id]
        # 如果該類別照片超過 200 張，就隨機抽取 200 張；如果不夠 200 張，就全拿
        num_to_sample = min(samples_per_class, len(indices))
        sampled_idx = random.sample(indices, num_to_sample)
        selected_indices.extend(sampled_idx)
        print(f"  - {class_name}: 原始 {len(indices)} 張 -> 抽取 {num_to_sample} 張")

    # 建立平衡後的新資料集 (總數應為 5 * 200 = 1000 張)
    balanced_dataset = Subset(full_dataset, selected_indices)

    # 4. 切分訓練集 (80%) 與驗證集 (20%)
    train_size = int(0.8 * len(balanced_dataset))
    val_size = len(balanced_dataset) - train_size
    train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])

    # 5. 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes


if __name__ == '__main__':
    try:
        train_loader, val_loader, classes = get_data_loaders()
        print("\n🎉 恭喜！資料集載入與平衡成功！")
        print(f"📦 類別列表：{classes}")
        print(f"📚 最終用於訓練的照片數量：{len(train_loader.dataset)} 張")
        print(f"📝 最終用於考試(驗證)的照片數量：{len(val_loader.dataset)} 張")

    except Exception as e:
        print("❌ 發生錯誤，請檢查資料夾路徑。")
        print(f"錯誤訊息：{e}")