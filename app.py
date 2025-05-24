import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision.models import vgg16
import torch.nn as nn

#Page Config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered", page_icon="🧠")

#Sidebar
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=150)
    st.markdown("## 🤖 Model Info")
    st.markdown("""
    - **Model**: VGG16 (Transfer Learning)
    - **Framework**: PyTorch
    - **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
    - **Classes**: Pituitary, Glioma, No Tumor, Meningioma
    - **Source**: [Hugging Face](https://huggingface.co/mtalhazafar/brain-tumor-detection-model)
    """)
    st.markdown("Made with ❤️ by [@MTalhaZafar](https://huggingface.co/mtalhazafar)")

#Title & Description
st.title("🧠 Brain Tumor Detection System")
st.markdown("Upload a **brain MRI image** to detect if there is a tumor and its type using a deep learning model trained with transfer learning on brain scans.")

#Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="mtalhazafar/brain-tumor-detection-model",
        filename="model.pth"
    )
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 4)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_labels = ['Pituitary', 'Glioma', 'No Tumor', 'Meningioma']

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Image Upload
uploaded_file = st.file_uploader("📤 Choose a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼 Uploaded Image', use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing the MRI image..."):
            img = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img)
                probabilities = torch.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                confidence = probabilities[predicted].item()

            label = class_labels[predicted.item()]
            result = "🟢 **No Tumor Detected**" if label == "No Tumor" else f"🔴 **Tumor Type:** {label}"

        #Display Result
        st.success(f"🎯 Prediction Result: {result}")
        st.info(f"🧪 Confidence: `{confidence*100:.2f}%`")

        #Show All Class Probabilities
        st.subheader("📊 Class Probabilities")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_labels[i]}: `{prob * 100:.2f}%`")
else:
    st.warning("Please upload an MRI image to get started.")
