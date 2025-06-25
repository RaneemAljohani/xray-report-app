import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

st.set_page_config(page_title="Chest X-ray Auto Report Generator", layout="wide")
st.title("🩻 Chest X-ray Auto Report Generator")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("nathansutton/generate-cxr")
    model = BlipForConditionalGeneration.from_pretrained("nathansutton/generate-cxr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("📝 Enter clinical context or prompt", value="Indication: Evaluate for infection, trauma, or foreign body in pediatric chest.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🔍 Generate Detailed Report"):
        with st.spinner("Analyzing X-ray and generating report..."):
            inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=True).to(device)
            output = model.generate(**inputs, max_new_tokens=512)
            report = processor.decode(output[0], skip_special_tokens=True)

        st.subheader("📋 Generated Detailed Report")
        st.markdown(f"""
### 🩻 Image Overview
- Projection: Frontal chest X-ray.
- Patient Positioning: Supine/semi-upright (likely pediatric).

### 🔍 Systematic Analysis

#### Findings:
- {report}

---

🧠 Impression:  
Please correlate clinically and consider additional investigations if needed.
""")

