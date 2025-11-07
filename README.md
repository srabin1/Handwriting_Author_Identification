# Handwriting Author Identification (CSAFE Handwriting Dataset)

This project trains convolutional neural networks (CNNs) to perform **writer identification** — determining **which person wrote a given handwriting sample**.  
Applications include **check fraud detection**, **forensics**, and **signature verification**.

---

## 📦 Dataset Source

CSAFE Handwriting Database (public, forensic-oriented):  
https://data.csafe.iastate.edu/HandwritingDatabase/

This dataset includes:
- 2,430 handwriting samples
- ~475 unique writers
- Multiple sessions & prompts per writer (letters, words, paragraphs, digits)

---

## 📁 Repository Structure

| Folder / File     | Purpose | Plain Explanation |
|------------------|---------|------------------|
| `data/writers/`  | Each writer’s images grouped by writer ID | Needed so model learns each person’s handwriting style |
| `splits/`        | `train.json`, `val.json`, `test.json` | Like assigning students to class, practice exam, and final exam |
| `checkpoints/`   | Saved trained models | Saves the “learned brain” so training doesn’t restart |
| `src/`           | Training / preprocessing scripts | The code that runs the experiments |

---

## ✋ Why We Group Images by Writer

All handwriting from the same writer must be grouped under a single writer ID (e.g., `w0001`).

Example filename:  
w0001_s03_pLN_D_r02.png
^---- Writer ID (this is the class label)

If images are not grouped by writer →  
The model cannot learn each person’s characteristic handwriting patterns:  
- Slant & curvature  
- Stroke smoothness  
- Letter formation habits  
- Baseline alignment  
- Pen pressure rhythm  

Grouping enforces **consistent identity clusters**, enabling the CNN to learn style instead of content.

---

## 🚫 Why We Do **Not** Split by Image (Important!)

We split **by writer**, not by image.

| Bad: Split by image | Good: Split by writer |
|---------------------|----------------------|
| The model sees samples of the same writer in both train and test → **fake high accuracy** | No writer appears in both train and test → **real evaluation** |

This prevents **identity leakage** and ensures **honest accuracy**.

---

## 🧱 Understanding the Filename Structure

| Part | Example | Meaning |
|------|---------|---------|
| Writer ID | `w0001` | Identity (this is the classification label) |
| Session | `s03` | Variation across time (do *not* treat as separate writers) |
| Prompt Type | `pLN`, `pW`, `pPH`, `pDIG` | The text being written — *not* used for classification |
| Region Code | `D`, `R`, `OZ` | Spatial region on page (background variability) |
| Repetition Number | `r02` | Multiple samples of same writer |

We classify **only by Writer ID**.

---

## 🗂️ Data Preparation

### 1️⃣ Group images by writer

```bash
for f in data/AllHandwritingImages/*.png; do
  writer=$(basename "$f" | cut -d'_' -f1)
  mkdir -p "data/writers/$writer"
  mv "$f" "data/writers/$writer/"
done
```
---
### 2️⃣ Shuffle writers & create splits (PowerShell on Windows)
```
Get-ChildItem .\data\writers -Directory |
  Select-Object -ExpandProperty Name |
  Sort-Object { Get-Random } |
  Set-Content .\writers_shuffled.txt
```

Check writer count:
```
wc -l writers_shuffled.txt
```

$all   = Get-Content .\writers_shuffled.txt
$train = $all | Select-Object -First 333
$val   = $all | Select-Object -Skip 333 -First 71
$test  = $all | Select-Object -Skip 404

$train | Set-Content .\train.list
$val   | Set-Content .\val.list
$test  | Set-Content .\test.list

convert lists -> JSON:
```
New-Item -ItemType Directory -Path .\splits -Force | Out-Null
(Get-Content .\train.list) | ConvertTo-Json > .\splits\train.json
(Get-Content .\val.list)   | ConvertTo-Json > .\splits\val.json
(Get-Content .\test.list)  | ConvertTo-Json > .\splits\test.json
```

## 🧠 Training (ResNet18 Transfer Learning)

We use:

Grayscale → 3-channel replicate

ImageNet normalization

Label smoothing

AdamW optimizer

Training and inference code is located in: src/train_resnet_closedset.py

### ✅ Run the code: run CSAFE_Handwriting.ipynb 
