import os
import json
from pathlib import Path
import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
from doclayout_yolo import YOLOv10
import json, re
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF, ignoring image content.
    Returns a list of strings, where each string is the text from a page.
    """
    text_per_page = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            # Get text blocks. 'text' method by default tries to extract text only.
            # We are not explicitly looking for images here, just extracting text.
            text_per_page.append(page.get_text())
        doc.close()
        print(f"Successfully extracted text from {len(text_per_page)} pages of {pdf_path}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        text_per_page = [] # Return empty if there's an error
    return text_per_page

def convert_pdf_page_to_image(pdf_path, page_number, dpi=200):
    """
    Converts a specific page of a PDF into a PIL Image.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)) # Render at 200 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Error converting page {page_number} of {pdf_path} to image: {e}")
        return None
    
def _is_title_like(name: str) -> bool:
    """Keep likely heading labels; exclude 'Page-header' etc."""
    n = name.lower().replace("_", "-").strip()
    if "page" in n and "header" in n:
        return False
    if "title" in n:
        return True
    if "section" in n and "header" in n:
        return True
    if "heading" in n:
        return True
    return False

def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    t = re.sub(r"^[•\-\u2022\.\s]+", "", t)
    return t

def _cluster_levels_by_height(cands):
    """
    cands: list of dicts with key 'h_px'
    Returns a list of level strings for each candidate: 'H1'/'H2'/'H3'.
    """
    if not cands:
        return []

    K = min(3, len(cands))
    heights = np.array([[c["h_px"]] for c in cands], dtype=float)

    if _HAS_SKLEARN and K >= 2:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(heights)
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)[::-1]  # largest height first
    else:
        # Fallback: quantile buckets
        if K == 1:
            labels = np.zeros(len(cands), dtype=int)
            order = [0]
        else:
            qs = np.quantile(heights.ravel(), np.linspace(0, 1, K + 1))
            labels = np.zeros(len(cands), dtype=int)
            for i, h in enumerate(heights.ravel()):
                for b in range(K):
                    if qs[b] <= h <= qs[b + 1]:
                        labels[i] = min(b, K - 1)
                        break
            means = [np.mean(heights.ravel()[labels == b]) for b in range(K)]
            order = np.argsort(means)[::-1]

    cluster_to_level = {order[i]: f"H{i+1}" for i in range(K)}
    return [cluster_to_level[int(lab)] for lab in labels]

def process_pdfs():
    # Get input and output directories
    input_dir = "/app/input"
    output_image_dir = "/app/output/processed_images"
    output_dir = "/app/output"
    model_path = "/app/model/model.pt"

    # Create output directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    DPI = 200
    _SCALE = DPI / 72.0  # px per PDF point


    if not os.path.exists(model_path):
        print(f"Error: model.pt not found at {os.path.abspath(model_path)}")
        print("Please ensure 'model.pt' is in the same directory as this notebook.")
    else:
        print(f"Loading DocLayout-YOLO model from {model_path}...")
        model = YOLOv10(model_path)
        print("DocLayout-YOLO model loaded.")

        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in '{input_dir}'. Please place your PDFs in this directory.")
        else:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_dir, pdf_file)
                print(f"\n--- Processing PDF: {pdf_file} ---")

                # Step 1: Parse PDF and ignore images (text extraction)
                extracted_text = extract_text_from_pdf(pdf_path)
                for i, page_text in enumerate(extracted_text):
                    print(f"Page {i+1} Text (first 200 chars):\n{page_text[:200]}...\n")
                    # You can save this text to a file if needed, e.g.:
                    # with open(os.path.join(output_image_directory, f"{pdf_file}_page_{i+1}_text.txt"), "w", encoding="utf-8") as f:
                    #     f.write(page_text)

                # Step 2: Process parsed PDFs (pages) with the pretrained DocLayout-YOLO model
                doc = fitz.open(pdf_path)
                for page_num in range(doc.page_count):
                    print(f"Processing page {page_num + 1} for DocLayout-YOLO...")
                    pil_image = convert_pdf_page_to_image(pdf_path, page_num)

                    if pil_image:
                        # Convert PIL Image to OpenCV format (numpy array)
                        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        temp_image_path = os.path.join(output_image_dir, f"temp_{pdf_file}_page_{page_num + 1}.jpg")
                        cv2.imwrite(temp_image_path, opencv_image)

                        # Perform prediction
                        try:
                            det_res = model.predict(
                                temp_image_path,
                                imgsz=1024,
                                conf=0.2,
                            )

                            # Annotate and save the result
                            annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
                            result_image_path = os.path.join(output_image_dir, f"result_{os.path.splitext(pdf_file)[0]}_page_{page_num + 1}.jpg")
                            cv2.imwrite(result_image_path, cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)) # Convert back to BGR for cv2.imwrite
                            print(f"DocLayout-YOLO result saved for {pdf_file} page {page_num + 1} to {result_image_path}")

                        except Exception as e:
                            print(f"Error during DocLayout-YOLO prediction for {pdf_file} page {page_num + 1}: {e}")
                        finally:
                            # Clean up temporary image
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                    else:
                        print(f"Skipping DocLayout-YOLO for {pdf_file} page {page_num + 1} due to image conversion error.")
                doc.close()

        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            print(f"\n=== Building outline for: {pdf_file} ===")

            doc = fitz.open(pdf_path)
            title_candidates = []

            try:
                for page_num in range(doc.page_count):  # page_num is 0-based
                    pil_image = convert_pdf_page_to_image(pdf_path, page_num, dpi=DPI)
                    if pil_image is None:
                        continue

                    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    temp_image_path = os.path.join(
                        output_image_dir, f"temp_{pdf_file}_page_{page_num}.jpg"
                    )
                    cv2.imwrite(temp_image_path, opencv_image)

                    try:
                        det_res = model.predict(temp_image_path, imgsz=1024, conf=0.2)
                        res = det_res[0]

                        names = getattr(res, "names", {}) or {}
                        if not names and hasattr(res, "boxes") and hasattr(res.boxes, "cls"):
                            uniq = np.unique(res.boxes.cls.cpu().numpy()).astype(int).tolist()
                            names = {i: str(i) for i in uniq}

                        wanted_ids = {i for i, n in names.items() if _is_title_like(str(n))}

                        if not hasattr(res, "boxes") or res.boxes is None or not len(res.boxes):
                            continue

                        xyxy = res.boxes.xyxy.cpu().numpy().astype(float)
                        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                        confs  = res.boxes.conf.cpu().numpy().astype(float)

                        page_obj = doc[page_num]
                        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                            if cls_id not in wanted_ids:
                                continue

                            # YOLO px -> PDF points clip
                            rect = fitz.Rect(x1/_SCALE, y1/_SCALE, x2/_SCALE, y2/_SCALE)
                            txt = _clean_text(page_obj.get_text("text", clip=rect))
                            if not txt:
                                continue

                            h_px = float(y2 - y1)
                            area_px = float((x2 - x1) * (y2 - y1))
                            y_center = float(0.5 * (y1 + y2))

                            title_candidates.append({
                                "page": page_num,         # 0-based (as requested)
                                "text": txt,
                                "conf": float(conf),
                                "x1_px": float(x1), "y1_px": float(y1),
                                "x2_px": float(x2), "y2_px": float(y2),
                                "h_px": h_px,
                                "area_px": area_px,
                                "y_center_px": y_center,
                                "label_name": str(names.get(int(cls_id), cls_id)),
                            })

                    except Exception as e:
                        print(f"Prediction error on {pdf_file} page {page_num}: {e}")
                    finally:
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
            finally:
                doc.close()

            # Deduplicate (page, text) keeping highest confidence — page is 0-based now
            dedup = {}
            for c in title_candidates:
                key = (c["page"], c["text"].lower())
                if key not in dedup or c["conf"] > dedup[key]["conf"]:
                    dedup[key] = c
            candidates = list(dedup.values())

            doc_title = ""
            outline_items = []

            if candidates:
                # === Select document title: globally largest bbox height; tie-break by higher conf,
                # then earliest page, then top-most on the page ===
                title_cand = sorted(
                    candidates,
                    key=lambda c: (c["h_px"], c["conf"], -c["page"], -c["y1_px"]),
                    reverse=True
                )[0]
                doc_title = title_cand["text"]

                # Exclude this item from outline (do not include in H1)
                candidates = [
                    c for c in candidates
                    if not (c["page"] == title_cand["page"] and c["text"] == title_cand["text"])
                ]

            if candidates:
                # Cluster remaining candidates and assign H1/H2/H3
                levels = _cluster_levels_by_height(candidates)
                for c, lvl in zip(candidates, levels):
                    c["level"] = lvl

                # Sort by (page asc, vertical position asc), pages are 0-based
                candidates.sort(key=lambda c: (c["page"], c["y1_px"]))

                outline_items = [
                    {"level": c["level"], "text": c["text"], "page": c["page"]}
                    for c in candidates
                ]

            out = {"title": doc_title, "outline": outline_items}
            out_path = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=4)

            print(f"Wrote outline with {len(outline_items)} items to: {out_path}")



if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs() 
    print("completed processing pdfs")