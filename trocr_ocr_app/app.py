import streamlit as st
from PIL import Image
from handwritten_classifier import is_handwritten

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="Smart OCR with TrOCR", layout="centered")
st.title("🧠 Smart OCR using Microsoft TrOCR")

uploaded_file = st.file_uploader("Upload an image (printed or handwritten)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Detecting content type...")
    if is_handwritten(image):
        model_type = "handwritten"
        st.success("Detected: ✍️ Handwritten Text")
    else:
        model_type = "printed"
        st.success("Detected: 🖨️ Printed Text")

    # Load appropriate model
    model_name = f"microsoft/trocr-base-{model_type}"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    st.write("🔄 Running OCR...")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("📄 Extracted Text:")
    st.text_area("OCR Output", generated_text, height=200)
