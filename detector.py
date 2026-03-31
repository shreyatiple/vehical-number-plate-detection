import cv2
import numpy as np
import easyocr
import re
import streamlit as st


@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)


def preprocess_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    versions = {}

    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions["otsu"] = thresh

    adapt = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    versions["adaptive"] = adapt

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    versions["clahe"] = clahe.apply(gray)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    versions["sharp"] = cv2.filter2D(gray, -1, kernel)

    return versions, gray


def find_plate_candidates(gray):
    candidates = []
    h, w = gray.shape

    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) >= 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            aspect = cw / ch if ch > 0 else 0
            if 1.5 <= aspect <= 6.0 and cw > 60 and ch > 15:
                pad = 5
                candidates.append((
                    max(0, x - pad), max(0, y - pad),
                    min(w, x + cw + pad), min(h, y + ch + pad),
                    cv2.contourArea(cnt)
                ))

    # Fallback strip regions
    for region in [
        (0, int(h * 0.4), w, int(h * 0.7)),
        (0, int(h * 0.55), w, int(h * 0.85)),
        (int(w * 0.1), int(h * 0.5), int(w * 0.9), int(h * 0.9)),
    ]:
        candidates.append((*region, 0))

    return candidates


def clean_plate_text(text):
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\s\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(text.split())


def score_plate_text(text):
    if not text or len(text) < 4:
        return 0
    score = 0
    cleaned = text.replace(' ', '').replace('-', '')
    if 5 <= len(cleaned) <= 12:
        score += 30
    if any(c.isalpha() for c in cleaned) and any(c.isdigit() for c in cleaned):
        score += 40
    if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$', cleaned):
        score += 30
    elif re.match(r'^[A-Z0-9]{4,10}$', cleaned):
        score += 10
    return score


def detect_plates(img_pil, reader, sensitivity=3):
    """
    Main entry point.
    img_pil  : PIL Image (RGB)
    reader   : EasyOCR Reader instance
    sensitivity : int 1-5

    Returns list of dicts sorted by best match first:
        plate, confidence, score, region (x1,y1,x2,y2), method
    """
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    preprocessed, gray = preprocess_image(img_bgr)
    candidates = find_plate_candidates(gray)
    min_conf = max(0.1, 0.5 - (sensitivity - 1) * 0.1)

    results = []
    seen = set()

    for (x1, y1, x2, y2, _) in candidates:
        for version_name, proc_img in preprocessed.items():
            crop = proc_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            if cw < 200:
                scale = max(2, 200 // cw)
                crop = cv2.resize(crop, (cw * scale, ch * scale),
                                  interpolation=cv2.INTER_CUBIC)
            try:
                ocr_out = reader.readtext(crop, detail=1, paragraph=False)
            except Exception:
                continue

            for (_, text, conf) in ocr_out:
                if conf < min_conf:
                    continue
                cleaned = clean_plate_text(text)
                if not cleaned or len(cleaned) < 4:
                    continue
                key = cleaned.replace(' ', '').replace('-', '')
                if key in seen:
                    continue
                sc = score_plate_text(cleaned)
                if sc < 20:
                    continue
                seen.add(key)
                results.append({
                    "plate": cleaned,
                    "confidence": round(conf * 100, 1),
                    "score": sc,
                    "region": (x1, y1, x2, y2),
                    "method": version_name,
                })

    results.sort(key=lambda r: r["score"] + r["confidence"], reverse=True)
    return results


def draw_annotations(img_pil, results):
    """Draw bounding boxes on a copy of the image. Returns RGB numpy array."""
    annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    colors = [(0, 200, 100), (0, 120, 255), (180, 50, 200)]

    for i, res in enumerate(results[:3]):
        x1, y1, x2, y2 = res["region"]
        color = colors[i % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{res['plate']}  {res['confidence']}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - 10, lh + 5)
        cv2.rectangle(annotated, (x1, ty - lh - 6), (x1 + lw + 8, ty + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 4, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)