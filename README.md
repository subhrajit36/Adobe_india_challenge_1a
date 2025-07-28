# Challenge 1a: PDF Outline Extractor

## Goal
Extract hierarchical headings (H1, H2, H3) from PDFs and output a structured JSON outline.

## Approach
- Parse PDFs using PyMuPDF, excluding images.
- Detect title/header regions with [DocLayout‑YOLO](https://github.com/opendatalab/DocLayout-YOLO) (50MB).
- Split detected regions into individual lines.
- Cluster headings by font size using K-means.
- Exclude the largest-font line as the document "title".

## Features
- **Fast detection:** YOLOv10 (DocLayout‑YOLO).
- **Title selection:** largest-font line.
- **Heading hierarchy:** cluster by font size or bounding box height.
- **Line-level extraction:** split multi-line regions.
- **Cleanup:** trim whitespace, remove duplicates.
- **Fallback:** font size heuristics if no YOLO detections.
- **Structured output:** one JSON per PDF in `output_json/`.

## Project Structure
```
Challenge_1a/
├── PDFs/                # Input PDFs
├── model.pt             # DocLayout‑YOLO checkpoint
├── output_images/       # Annotated preview images (optional)
├── output_json/         # Outline JSONs
├── notebook.ipynb       # End-to-end runner
└── README.md            # This file
```

## Requirements
- Python 3.9+
- Dependencies: `PyMuPDF`, `opencv-python`, `Pillow`, `doclayout-yolo`, `scikit-learn` (optional: `huggingface_hub`)

## Logic Overview
1. Detect layout regions with DocLayout‑YOLO.
2. Filter heading-like classes.
3. Split regions into lines.
4. Clean and deduplicate.
5. Select document "title" (largest font); exclude from outline.
6. Cluster lines by font size to assign H1/H2/H3.
7. Sort by page and vertical order; write JSON.
8. Fallback: use font-size heuristics if no detections.


## Refrences
- DocLayout‑YOLO for region detection.
- PyMuPDF for PDF parsing.


---
