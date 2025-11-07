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
