
import os, json, argparse, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

def list_images(writer_dir):
    exts={".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif"}
    return [str(p) for p in Path(writer_dir).glob("*") if p.suffix.lower() in exts]

class ImageListDataset(Dataset):
    def __init__(self, items, image_size=224, train=True):
        self.items = items  # list of (path, class_idx)
        if train:
            self.tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size,image_size)),
                transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05), shear=2),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
                transforms.ToTensor(), transforms.Normalize([0.5],[0.5]),
                transforms.RandomErasing(p=0.10, scale=(0.01,0.03), ratio=(0.3,3.3))
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(), transforms.Normalize([0.5],[0.5]),
            ])
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        path,y = self.items[i]
        x = self.tf(Image.open(path).convert("L"))
        return x, y

def make_resnet18_gray(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
    m.conv1.weight.data = w.mean(1,keepdim=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = Path(args.data_root)
    # classes = ALL writer folders present
    writers = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("w")])
    writer_to_idx = {w:i for i,w in enumerate(writers)}
    print("Writers (classes):", len(writers))

    # build image-level splits (per writer)
    random.seed(123)
    train_items, val_items, test_items = [], [], []
    for w in writers:
        imgs = sorted(list_images(data_root/w))
        if len(imgs)==0: continue
        random.shuffle(imgs)
        n = len(imgs)
        n_tr = max(1, int(0.70*n))
        n_val = max(1, int(0.15*n))
        n_te = max(1, n - n_tr - n_val)
        y = writer_to_idx[w]
        train_items += [(p,y) for p in imgs[:n_tr]]
        val_items   += [(p,y) for p in imgs[n_tr:n_tr+n_val]]
        test_items  += [(p,y) for p in imgs[n_tr+n_val:]]
    print("Images -> train/val/test:", len(train_items), len(val_items), len(test_items))

    # datasets/loaders
    tr_ds = ImageListDataset(train_items, args.image_size, True)
    va_ds = ImageListDataset(val_items,   args.image_size, False)
    te_ds = ImageListDataset(test_items,  args.image_size, False)
    num_workers = 0
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model/opt/sched
    model = make_resnet18_gray(len(writers)).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_val = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    def loop(dl, train):
        model.train() if train else model.eval()
        tot=correct=0; loss_sum=0.0; n_batches=0
        with torch.set_grad_enabled(train):
            for x,y in dl:
                x,y = x.to(device), y.to(device)
                if train: opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                    logits = model(x)
                    loss = crit(logits,y)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                pred = logits.argmax(1)
                correct += (pred==y).sum().item(); tot += y.numel()
                loss_sum += loss.item(); n_batches += 1
        return loss_sum/max(1,n_batches), correct/max(1,tot)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = loop(tr_dl, True)
        va_loss, va_acc = loop(va_dl, False)
        sch.step(va_acc)
        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model_state": model.state_dict(), "writer_to_idx": writer_to_idx},
                       os.path.join(args.ckpt_dir, "best_closedset.pt"))

    # test with best
    ckpt = torch.load(os.path.join(args.ckpt_dir,"best_closedset.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc = loop(te_dl, False)
    print(f"[TEST] loss {te_loss:.4f} acc {te_acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",   type=str, required=True)
    ap.add_argument("--ckpt_dir",    type=str, required=True)
    ap.add_argument("--image_size",  type=int, default=224)
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay",type=float, default=1e-4)
    args = ap.parse_args()
    run(args)
