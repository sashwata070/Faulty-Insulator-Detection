import os
import shutil
import random

# Force CPU for preprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # <-- Hide GPU from PyTorch
print("Preprocessing (CAE + Wiener + BMP) → Running on CPU")

input_img_dir = "/content/drive/MyDrive/training-dataset"
input_lbl_dir = input_img_dir  # labels are in same dir
output_base = "/content/project/imagefolder"

class_names = ['normal_pin', 'defective_pin', 'normal_disc', 'defective_disc']
os.makedirs(output_base, exist_ok=True)
for cls in class_names:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

# Move images based on YOLO label
for img in os.listdir(input_img_dir):
    if img.endswith(".jpg"):
        label_path = os.path.join(input_lbl_dir, img.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_line = f.readline().strip()
                if label_line:
                    class_id = int(label_line.split()[0])
                    shutil.copy(os.path.join(input_img_dir, img), os.path.join(output_base, class_names[class_id], img))


from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import os
import sys
import time

# --- CAE / CLAHE Enhancement ---
def cae_enhance(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert to LAB color space (works better for contrast enhancement)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split into L, A, B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE only on L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # Merge back
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


# --- Paths ---
input_folder = "/content/project/imagefolder"
output_folder = "/content/project/cae"
os.makedirs(output_folder, exist_ok=True)

# --- Simple loading spinner ---
def loading_spinner(message, duration=1.5):
    spinner = ["|", "/", "-", "\\"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{message} {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # clear line


# --- Walk through subfolders ---
for root, dirs, files in os.walk(input_folder):
    if files:  # process only if folder has images
        rel_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_folder, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        # Show spinner while processing the folder
        loading_spinner(f"Processing folder: {rel_path}")

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                enhanced = cae_enhance(img)

                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, enhanced)

        print(f"✅ Finished folder: {rel_path}")

print("\n🎉 All images enhanced with CAE (CLAHE) and saved in:", output_folder)


import cv2
import numpy as np
import os
import sys
import time
from scipy.signal import wiener

# --- Wiener filter enhancement ---
def wiener_enhance(img):
    img = img.astype(np.float32) / 255.0  # scale to [0,1]
    enhanced = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        enhanced[:, :, c] = wiener(img[:, :, c])
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)

# --- Paths ---
input_folder = "/content/project/cae"
output_folder = "/content/project/cae_wiener"

# --- Simple loading spinner ---
def loading_spinner(message, duration=1.5):
    spinner = ["|", "/", "-", "\\"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{message} {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # clear line

# --- Walk through subfolders ---
for root, dirs, files in os.walk(input_folder):
    if files:  # process only if folder has images
        rel_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_folder, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        # Show spinner while processing the folder
        loading_spinner(f"Processing folder: {rel_path}")

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                enhanced = wiener_enhance(img_rgb)

                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

        print(f"✅ Finished folder: {rel_path}")

print("\n🎉 All images enhanced with Wiener filter and saved in:", output_folder)


**BMP**

from PIL import Image
import os
import sys
import time

# --- Simple loading spinner ---
def loading_animation(folder_name, total_files):
    spinner = ['|', '/', '-', '\\']
    for i in range(total_files * 2):  # twice the files (just for smooth effect)
        sys.stdout.write(f"\r⏳ Converting '{folder_name}' {spinner[i % 4]}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r✅ Finished converting folder: " + folder_name + " " * 20 + "\n")

# --- Paths ---
input_folder = "/content/project/cae_wiener"       # JPG dataset root
output_folder = "/content/project/imagefolder_bmp"  # BMP dataset root

# --- Convert all JPG/JPEG images inside class folders ---
for root, dirs, files in os.walk(input_folder):
    # Preserve relative class subfolder structure
    rel_path = os.path.relpath(root, input_folder)
    save_dir = os.path.join(output_folder, rel_path)
    os.makedirs(save_dir, exist_ok=True)

    # Filter only images
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]

    if not image_files:
        continue

    # --- Loading animation per subfolder (in background while saving) ---
    from threading import Thread
    anim_thread = Thread(target=loading_animation, args=(rel_path, len(image_files)))
    anim_thread.start()

    # --- Process all images in this subfolder ---
    for file in image_files:
        img_path = os.path.join(root, file)
        img = Image.open(img_path).convert("RGB")  # ensure RGB format

        bmp_name = os.path.splitext(file)[0] + ".bmp"
        bmp_path = os.path.join(save_dir, bmp_name)

        img.save(bmp_path)

    anim_thread.join()  # wait until animation finishes

print("🎉 All JPG images converted to BMP (with subfolders preserved).")


**# Switch to GPU**

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Now enable GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training → Using device: {device}")

from torchvision import models, transforms
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn.functional as F  # For MixUp and Focal Loss

# --- Config ---
batch_size = 32
epochs = 40  # Increased for potentially better convergence
num_classes = 4
patience = 11   # Tightened patience for early stopping
best_val_acc = 0.0
counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms (Added more advanced augmentations) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, shear=10),  # Added shear for perspective
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Simulate degradation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Dataset ---
dataset = ImageFolder("/content/project/imagefolder_bmp", transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# --- EfficientNet Model (Sticking with B0, but you can change to B2) ---
eff_model = models.efficientnet_b0(pretrained=True)

# Unfreeze all layers for fine-tuning
for param in eff_model.parameters():
    param.requires_grad = True

# Modify classifier (Increased dropout)
eff_model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(eff_model.classifier[1].in_features, num_classes)
)

eff_model = eff_model.to(device)

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

criterion_eff = FocalLoss()  # Replaced with Focal Loss (remove label_smoothing if using this)

# --- Optimizer, Scheduler ---
optimizer_eff = optim.AdamW(eff_model.parameters(), lr=0.0001)  # Lowered LR for finer tuning
scheduler_eff = ReduceLROnPlateau(optimizer_eff, 'max', patience=3, factor=0.5)

# --- MixUp Function ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Training Loop with EarlyStopping and MixUp ---
for epoch in range(epochs):
    eff_model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Apply MixUp
        images, targets_a, targets_b, lam = mixup_data(images, labels)

        outputs = eff_model(images)
        loss = mixup_criterion(criterion_eff, outputs, targets_a, targets_b, lam)

        optimizer_eff.zero_grad()
        loss.backward()
        optimizer_eff.step()

        # For accuracy, use original labels (approximate)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()  # Note: This is approx due to MixUp
        total += labels.size(0)

    # --- Validation ---
    eff_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = eff_model(images)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    val_acc = accuracy_score(y_true, y_pred)
    scheduler_eff.step(val_acc)
    train_acc = correct / total

    print(f"[EfficientNet-B0] Epoch {epoch+1}/{epochs} - "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # --- EarlyStopping ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(eff_model.state_dict(), "best_effnet.pth")  # save best model
        print("✅ Validation improved, model saved.")
    else:
        counter += 1
        print(f"⚠️ No improvement for {counter} epoch(s).")
        if counter >= patience:
            print("⏹️ Early stopping triggered!")
            break

print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# ===============================
# ConvNeXt-Tiny Training for Enhanced Insulator Images
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time

# ---- Dataset Path ----
data_dir = "/content/project/imagefolder_bmp"

# ---- Transforms ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Dataset Split (90% train, 10% val) ----
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# ---- Model ----
model = models.convnext_tiny(weights="IMAGENET1K_V1")
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, len(dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---- Loss, Optimizer, Scheduler ----
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.3)

# ---- Training ----
best_acc = 0.0
epochs = 40
patience = 11
no_improve = 0

for epoch in range(epochs):
    start = time.time()
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(f"[ConvNeXt-Tiny] Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Acc: {val_acc*100:.2f}% | Time: {time.time()-start:.1f}s")

    # ---- Early Stopping ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_convnext_tiny.pth")
        no_improve = 0
        print("✅ Validation improved, model saved.")
    else:
        no_improve += 1
        print(f"⚠️ No improvement for {no_improve} epoch(s).")
        if no_improve >= patience:
            print("⏹️ Early stopping triggered!")
            break

print(f"Best Validation Accuracy: {best_acc*100:.2f}%")


!pip install -U timm


# ===============================
# CoAtNet-0 Training for Enhanced Insulator Images
# ===============================

!pip install --upgrade timm -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import time

# ---- Dataset Path ----
data_dir = "/content/project/imagefolder_bmp"

# ---- Transforms ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Dataset Split ----
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# ---- Model ----
model = timm.create_model("coatnet_0_rw_224.sw_in1k", pretrained=True, num_classes=len(dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---- Loss, Optimizer, Scheduler ----
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.3)

# ---- Training ----
best_acc = 0.0
epochs = 40
patience = 11
no_improve = 0

for epoch in range(epochs):
    start = time.time()
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(f"[CoAtNet-0] Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Acc: {val_acc*100:.2f}% | Time: {time.time()-start:.1f}s")

    # ---- Early Stopping ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_coatnet0.pth")
        no_improve = 0
        print("✅ Validation improved, model saved.")
    else:
        no_improve += 1
        print(f"⚠️ No improvement for {no_improve} epoch(s).")
        if no_improve >= patience:
            print("⏹️ Early stopping triggered!")
            break

print(f"Best Validation Accuracy: {best_acc*100:.2f}%")

!pip install -q albumentations torch-optimizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# --- Config ---
batch_size = 32
epochs = 40
num_classes = 4 # Based on the class_names defined earlier in the notebook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Dataset (Assuming the preprocessed data is in /content/project/cae_wiener) ---
dataset = ImageFolder("/content/project/cae_wiener", transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# --------------------------------------------------------------
#  Optimized CNN (your architecture)
# --------------------------------------------------------------
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimizedCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.SiLU(), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

# --------------------------------------------------------------
#  Model + Loss + SGD Optimizer
# --------------------------------------------------------------
cnn_model = OptimizedCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(
    cnn_model.parameters(),
    lr=0.05,           # Good starting LR for SGD on CNNs
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True      # Nesterov momentum = better convergence
)

scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# --------------------------------------------------------------
#  Early Stopping Setup
# --------------------------------------------------------------
patience = 11
best_val_acc = 0.0
patience_counter = 0
best_model_path = "best_cnn_sgd.pth"

# --------------------------------------------------------------
#  Training Loop with Early Stopping
# --------------------------------------------------------------
print("Starting training with SGD + Early Stopping (patience=9)...\n")

for epoch in range(epochs):
    # -------------------- Training --------------------
    cnn_model.train()
    train_correct = 0
    train_total = 0
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # Step scheduler once per epoch
    scheduler.step()

    train_acc = train_correct / train_total
    avg_loss = running_loss / len(train_loader)

    # -------------------- Validation --------------------
    cnn_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    val_acc = accuracy_score(y_true, y_pred)

    # -------------------- Logging ----
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # -------------------- Early Stopping Logic --------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(cnn_model.state_dict(), best_model_path)
        print(f"   Validation improved → model saved ({best_val_acc:.4f})")
    else:
        patience_counter += 1
        print(f"   No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after epoch {epoch+1}!")
            break

# --------------------------------------------------------------
#  Load Best Model & Final Test-Time Evaluation
# --------------------------------------------------------------
cnn_model.load_state_dict(torch.load(best_model_path))
cnn_model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

final_acc = accuracy_score(y_true, y_pred)
print(f"\nBEST VALIDATION ACCURACY: {final_acc:.4f} (saved model)")

# Optional: Test-Time Augmentation (horizontal flip)
def tta_val(loader):
    cnn_model.eval()
    preds = []
    flip = torch.flip
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            out1 = cnn_model(images)
            out2 = cnn_model(flip(images, dims=[3]))  # horizontal flip
            out = (out1 + out2) / 2
            _, pred = torch.max(out, 1)
            preds.extend(pred.cpu().numpy())
    return preds

tta_pred = tta_val(val_loader)
tta_acc = accuracy_score(y_true, tta_pred)
print(f"With simple TTA (hflip): {tta_acc:.4f}")

# --------------------------------------------------------------
#  1. Imports & Device
# --------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import timm
from torchvision import models
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------
#  2. Load your validation DataLoader
# --------------------------------------------------------------
# Example (replace with your own val_loader from training cell)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder("/content/project/imagefolder_bmp", transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# --------------------------------------------------------------
#  3. Define Your Custom CNN (OptimizedCNN)
# --------------------------------------------------------------
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimizedCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.SiLU(), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

# --------------------------------------------------------------
#  4. Load the three trained models
# --------------------------------------------------------------
num_classes = 4
class_names = ["normal_pin", "defective_pin", "normal_disc", "defective_disc"]

# ---- EfficientNet-B0 ------------------------------------------------
effnet = models.efficientnet_b0(pretrained=False)
effnet.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(effnet.classifier[1].in_features, num_classes)
)
effnet.load_state_dict(torch.load("best_effnet.pth", map_location=device))
effnet = effnet.to(device).eval()

# ---- CoATNet ------------------------------------------------
coatnet = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=num_classes)
coatnet.load_state_dict(torch.load("best_coatnet0.pth", map_location=device))
coatnet = coatnet.to(device).eval()

# ---- CUSTOM CNN (Replaces ConvNeXt) ---------------------------------
cnn = OptimizedCNN(num_classes=num_classes)
cnn.load_state_dict(torch.load("best_cnn_sgd.pth", map_location=device))  # <--- YOUR CNN PATH
cnn = cnn.to(device).eval()

# Update models list
models = [effnet, coatnet, cnn]
model_names = ["EffNet-B0", "CoATNet", "CustomCNN"]

# --------------------------------------------------------------
#  5. Inference + Majority Voting (≥2 = Faulty)
# --------------------------------------------------------------
all_true, all_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        # Forward pass
        logits_list = [m(images) for m in models]
        preds_each = [torch.argmax(logits, dim=1) for logits in logits_list]
        preds_stack = torch.stack(preds_each)  # (3, B)

        # Count votes for faulty (class 1 or 3)
        is_faulty_each = torch.isin(preds_stack, torch.tensor([1, 3], device=device))
        votes_faulty = is_faulty_each.sum(dim=0)  # (B,)

        ensemble_pred = torch.zeros_like(labels)
        for b in range(images.size(0)):
            if votes_faulty[b] >= 2:
                # Majority says faulty → pick type by avg logit
                avg_faulty = torch.stack([logits_list[m][b, [1, 3]] for m in range(3)]).mean(0)
                ensemble_pred[b] = 1 if avg_faulty[0] > avg_faulty[1] else 3
            else:
                # Normal → pick by avg logit
                avg_normal = torch.stack([logits_list[m][b, [0, 2]] for m in range(3)]).mean(0)
                ensemble_pred[b] = 0 if avg_normal[0] > avg_normal[1] else 2

        all_true.extend(labels.cpu().numpy())
        all_pred.extend(ensemble_pred.cpu().numpy())

# --------------------------------------------------------------
#  6. Evaluation
# --------------------------------------------------------------
overall_acc = accuracy_score(all_true, all_pred)
print(f"\n=== ENSEMBLE (EffNet + CoATNet + CustomCNN) ===")
print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

# Per-class
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    true_i = np.array(all_true) == i
    if true_i.sum() > 0:
        acc_i = accuracy_score(np.array(all_true)[true_i], np.array(all_pred)[true_i])
        print(f"  {name}: {acc_i:.4f}")
    else:
        print(f"  {name}: No samples")

# Faulty vs Normal
is_faulty_gt = np.isin(all_true, [1, 3])
is_faulty_pred = np.isin(all_pred, [1, 3])
tp = np.sum(is_faulty_gt & is_faulty_pred)
fp = np.sum(~is_faulty_gt & is_faulty_pred)
fn = np.sum(is_faulty_gt & ~is_faulty_pred)
tn = np.sum(~is_faulty_gt & ~is_faulty_pred)

print(f"\nFaulty-vs-Normal:")
print(f"  TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
print(classification_report(all_true, all_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Ensemble: EffNet + CoATNet + CustomCNN")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()
