
import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

# ---------- utils ----------
def load_split(path):
    with open(path, "r") as f:
        return json.load(f)

def list_images(writer_dir):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    return [p for p in Path(writer_dir).glob("*") if p.suffix.lower() in exts]

# ---------- dataset ----------
class WritersDataset(Dataset):
    def __init__(self, data_root, writer_ids, writer_to_idx, image_size=224, train=True):
        self.samples = []
        for wid in writer_ids:
            for imgp in list_images(Path(data_root)/wid):
                self.samples.append((str(imgp), wid))
        if train:
            self.tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05), shear=2),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5]),
                transforms.RandomErasing(p=0.10, scale=(0.01,0.03), ratio=(0.3,3.3))
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5]),
            ])
        self.writer_to_idx = writer_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, wid = self.samples[i]
        img = Image.open(path).convert("L")
        x = self.tf(img)
        y = self.writer_to_idx[wid]
        return x, y

# ---------- model ----------
def make_resnet18_gray(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # adapt first conv for 1-channel grayscale by averaging RGB weights
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight.data = w.mean(dim=1, keepdim=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------- training loop ----------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load splits
    train_ids = load_split(Path(args.splits_dir)/"train.json")
    val_ids   = load_split(Path(args.splits_dir)/"val.json")
    test_ids  = load_split(Path(args.splits_dir)/"test.json")

    # label map
    all_ids = sorted(set(train_ids + val_ids + test_ids))
    writer_to_idx = {w:i for i,w in enumerate(all_ids)}
    num_classes = len(writer_to_idx)
    print(f"Writers: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} total={num_classes}")

    # datasets/loaders
    train_ds = WritersDataset(args.data_root, train_ids, writer_to_idx, args.image_size, train=True)
    val_ds   = WritersDataset(args.data_root, val_ids,   writer_to_idx, args.image_size, train=False)
    test_ds  = WritersDataset(args.data_root, test_ids,  writer_to_idx, args.image_size, train=False)

    # Colab stability tip
    num_workers = 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model
    model = make_resnet18_gray(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    def loop(dl, train_mode):
        if train_mode: model.train()
        else: model.eval()
        total, correct, loss_sum, n_batches = 0,0,0.0,0
        with torch.set_grad_enabled(train_mode):
            for x,y in dl:
                x,y = x.to(device), y.to(device)
                if train_mode: optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                    logits = model(x)
                    loss = criterion(logits,y)
                if train_mode:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                loss_sum += loss.item()
                n_batches += 1
                pred = torch.argmax(logits, dim=1)
                correct += (pred==y).sum().item()
                total += y.numel()
        return loss_sum/max(n_batches,1), correct/max(total,1)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = loop(train_dl, True)
        val_loss, val_acc = loop(val_dl, False)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "writer_to_idx": writer_to_idx,
                "args": vars(args)
            }, os.path.join(args.ckpt_dir, "best_resnet18.pt"))

    # test
    ckpt = torch.load(os.path.join(args.ckpt_dir,"best_resnet18.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = loop(test_dl, False)
    print(f"[TEST] loss {test_loss:.4f} acc {test_acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",   type=str, required=True)
    ap.add_argument("--splits_dir",  type=str, required=True)
    ap.add_argument("--ckpt_dir",    type=str, required=True)
    ap.add_argument("--image_size",  type=int, default=224)
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--epochs",      type=int, default=15)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay",type=float, default=1e-4)
    args = ap.parse_args()
    run(args)
