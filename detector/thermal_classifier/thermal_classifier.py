# thermal_classifier.py  
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

data_dir = "/home/UFAD/zhou.zhuoyang/hacks/AIDER/data/generated_thermal_images"
batch_size = 64
epochs = 20
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

from collections import Counter
from pathlib import Path

print("\n[Sanity Check] ImageFolder classes:", full_dataset.classes)
print("[Sanity Check] class_to_idx:", full_dataset.class_to_idx)
print("[Sanity Check] total images in ImageFolder:", len(full_dataset))
print("[Sanity Check] targets Counter (all):", Counter(full_dataset.targets))

valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
fs_counts = {}
for cls in full_dataset.classes:
    cls_dir = Path(data_dir) / cls
    cnt = 0
    for p in cls_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            cnt += 1
    fs_counts[cls] = cnt
print("[Sanity Check] filesystem counts:", fs_counts)
print()  


indices = np.arange(len(full_dataset))
targets = np.array(full_dataset.targets)

try:
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=targets
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=targets[temp_idx]
    )
except ValueError:
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_ds = Subset(full_dataset, train_idx.tolist())
val_ds   = Subset(full_dataset, val_idx.tolist())
test_ds  = Subset(full_dataset, test_idx.tolist())

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0
assert len(train_idx) + len(val_idx) + len(test_idx) == len(full_dataset)

from collections import Counter
train_targets = targets[train_idx]
val_targets   = targets[val_idx]
test_targets  = targets[test_idx]
print("Total:", Counter(targets))
print("Train:", Counter(train_targets))
print("Val  :", Counter(val_targets))
print("Test :", Counter(test_targets))

os.makedirs("artifacts", exist_ok=True)
np.savez("artifacts/splits.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1); self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1); self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1); self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1); self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x))); x = F.relu(self.bn4(self.conv4(x))); x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x))); x = F.relu(self.bn6(self.conv6(x))); x = self.pool(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

model = DeeperCNN(num_classes=2).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0,0,0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits,y)
            loss_sum += loss.item()*x.size(0)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

best_val = -1.0
best_path = "artifacts/thermal_fire_classifier_best.pt"

for epoch in range(1,epochs+1):
    model.train()
    for x,y in train_loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), best_path)

model.load_state_dict(torch.load(best_path, map_location=device))

def evaluate_and_print_report(model, test_loader, device, class_to_idx):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    normal_idx = class_to_idx.get("thermal", 0)
    fire_idx   = class_to_idx.get("fire", 1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[fire_idx], average="binary",
        pos_label=fire_idx, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[normal_idx, fire_idx])

    print("=== Street-level Evaluation ===")
    print(f"Streets evaluated: {len(y_true)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(cm, "\n")

    remap = {normal_idx: 0, fire_idx: 1}
    y_true_r = np.vectorize(remap.get)(y_true)
    y_pred_r = np.vectorize(remap.get)(y_pred)
    print("Detailed Report:")
    print(classification_report(
        y_true_r, y_pred_r,
        target_names=["Normal", "Anomalous"], digits=4
    ))

test_loss, test_acc = evaluate(test_loader)
print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}\n")
evaluate_and_print_report(model, test_loader, device, class_to_idx=full_dataset.class_to_idx)

torch.save(model.state_dict(), "artifacts/thermal_fire_classifier_final.pt")
