import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import fitz  # PyMuPDF
import os
import tempfile

st.set_page_config(page_title="🧠 AI OCR | Handwritten & Printed", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>🧠 AI OCR App</h1>
    <p style='text-align: center;'>Upload an image or PDF (handwritten or printed), and let the AI extract the text!</p>
    <hr>
""", unsafe_allow_html=True)

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

model_type = st.radio("✍️ Text Type", ["Auto Detect", "Printed", "Handwritten"])

# --- Handwriting Detection Helper ---
def is_handwritten(image):
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    model_id = "microsoft/image-classification-base"
    classifier = AutoModelForImageClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = classifier(**inputs).logits
    predicted_class = logits.argmax(-1).item()
    return predicted_class == 1  # Assume 1 = handwritten (you can fine-tune this)

# --- Load Appropriate Model ---
@st.cache_resource
def load_model(model_choice):
    if model_choice == "handwritten":
        model_name = "microsoft/trocr-base-handwritten"
    else:
        model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

def pdf_to_images(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        doc = fitz.open(tmp_file.name)
        images = [Image.frombytes("RGB", page.get_pixmap().size, page.get_pixmap().samples) for page in doc]
        doc.close()
        os.remove(tmp_file.name)
        return images

def run_ocr(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if uploaded_file:
    ext = uploaded_file.name.lower().split('.')[-1]
    images = []

    if ext == "pdf":
        images = pdf_to_images(uploaded_file.read())
    else:
        image = Image.open(uploaded_file).convert("RGB")
        images = [image]

    for idx, image in enumerate(images):
        st.image(image, caption=f"Page {idx+1}", use_column_width=True)

        # Auto-detect
        if model_type == "Auto Detect":
            if is_handwritten(image):
                use_model = "handwritten"
                st.info("Detected: ✍️ Handwritten text")
            else:
                use_model = "printed"
                st.info("Detected: 🖨️ Printed text")
        elif model_type == "Printed":
            use_model = "printed"
            st.success("Selected: 🖨️ Printed text (manual)")
        else:
            use_model = "handwritten"
            st.success("Selected: ✍️ Handwritten text (manual)")

        try:
            processor, model = load_model(use_model)
            extracted_text = run_ocr(image, processor, model)
            st.markdown(f"#### 📄 Extracted Text:\n```{extracted_text}```")
        except Exception as e:
            st.error("❌ Failed to process text. Please check the image or model compatibility.")
            st.exception(e)
