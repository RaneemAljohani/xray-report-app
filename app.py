
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("nathansutton/generate-cxr")
    model = BlipForConditionalGeneration.from_pretrained("nathansutton/generate-cxr")
    return processor, model

processor, model = load_model()

st.title("📋 توليد تقارير أشعة الصدر باستخدام الذكاء الاصطناعي")
st.markdown("قم برفع صورة أشعة صدر (Chest X-ray) للحصول على تقرير طبي آلي.")

uploaded_file = st.file_uploader("🩻 ارفع صورة الأشعة هنا", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="صورة الأشعة المدخلة", use_column_width=True)

    prompt = "indication: evaluation for suspected pneumonia and fracture"

    if st.button("🔍 تحليل الصورة"):
        inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=True)
        out = model.generate(**inputs)
        report = processor.decode(out[0], skip_special_tokens=True)
        st.markdown("### 🧾 التقرير الناتج:")
        st.success(report)
