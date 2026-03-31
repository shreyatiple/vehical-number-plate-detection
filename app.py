import streamlit as st
from PIL import Image
from datetime import datetime
import pandas as pd

from detector import load_reader, detect_plates, draw_annotations

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Number Plate Detector",
    page_icon="🚗",
    layout="centered",
)

# ── Session state ────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    sensitivity = st.slider("Detection sensitivity", 1, 5, 3,
                            help="Higher = more candidates checked. Useful for tricky images.")
    show_boxes = st.toggle("Show bounding boxes", value=True)

    st.divider()
    st.caption("**Tips for best results**")
    st.caption("• Use a clear, well-lit photo")
    st.caption("• Plate should face the camera")
    st.caption("• Avoid heavy motion blur")

    if st.session_state.history:
        st.divider()
        st.caption("**Recent detections**")
        for item in reversed(st.session_state.history[-6:]):
            st.caption(f"🚘 `{item['plate']}` — {item['time']}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Number Plate Detector")
st.caption("Upload a vehicle image to extract its number plate using OpenCV + EasyOCR.")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Image", "Batch Processing"])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single image
# ────────────────────────────────────────────────────────────────────────────────
with tab_single:
    uploaded = st.file_uploader(
        "Upload a vehicle image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Detect Plate", use_container_width=True):
            with st.spinner("Analysing image…"):
                reader = load_reader()
                results = detect_plates(img, reader, sensitivity)

            if results:
                top = results[0]

                st.success("Plate detected!")
                st.markdown(f"## `{top['plate']}`")
                st.caption(f"Confidence: **{top['confidence']}%** · Method: {top['method']}")

                if show_boxes:
                    annotated = draw_annotations(img, results)
                    st.image(annotated, caption="Annotated", use_container_width=True)

                if len(results) > 1:
                    with st.expander(f"All candidates ({len(results)})"):
                        df = pd.DataFrame([
                            {"Plate": r["plate"], "Confidence %": r["confidence"], "Method": r["method"]}
                            for r in results
                        ])
                        st.dataframe(df, hide_index=True, use_container_width=True)

                # Save to history
                st.session_state.history.append({
                    "plate": top["plate"],
                    "confidence": top["confidence"],
                    "time": datetime.now().strftime("%H:%M"),
                })

                # Download
                result_txt = (
                    f"Plate      : {top['plate']}\n"
                    f"Confidence : {top['confidence']}%\n"
                    f"File       : {uploaded.name}\n"
                    f"Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                st.download_button("⬇ Download result", result_txt,
                                   "plate_result.txt", "text/plain")

            else:
                st.warning(
                    "No plate detected. Try increasing the sensitivity slider "
                    "or use a clearer image."
                )

# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch
# ────────────────────────────────────────────────────────────────────────────────
with tab_batch:
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if batch_files and st.button("🔍 Process All", use_container_width=True):
        reader = load_reader()
        rows = []
        prog = st.progress(0)

        for i, f in enumerate(batch_files):
            img_b = Image.open(f).convert("RGB")
            res = detect_plates(img_b, reader, sensitivity)
            plate = res[0]["plate"] if res else "Not detected"
            conf  = res[0]["confidence"] if res else "-"
            rows.append({"File": f.name, "Plate": plate,
                         "Confidence %": conf,
                         "Status": "✅" if res else "❌"})
            if res:
                st.session_state.history.append({
                    "plate": plate, "confidence": conf,
                    "time": datetime.now().strftime("%H:%M"),
                })
            prog.progress((i + 1) / len(batch_files))

        prog.empty()
        df_batch = pd.DataFrame(rows)
        st.dataframe(df_batch, hide_index=True, use_container_width=True)
        st.download_button("⬇ Download CSV", df_batch.to_csv(index=False),
                           "batch_results.csv", "text/csv")