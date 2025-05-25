# Brain Tumor Detection Using PyTorch (VGG16 + Transfer Learning)

Welcome to the **Brain Tumor Detection Using Deep Learning** project! This Python-based application leverages the power of deep learning and transfer learning (VGG16) to classify MRI brain scans as either tumor-affected or normal. The goal of this project is to assist in early detection of brain tumors using automated image classification. The model is trained and evaluated using a labeled dataset of brain MRI images and is deployed through a Streamlit interface for easy use.

## 🧠 Project Overview

The main objective is to leverage pre-trained deep learning models to detect brain tumors from MRI images. Here’s a quick breakdown:

1. **📁 Dataset & Preprocessing**
   The dataset includes labeled brain MRI images. Each image is resized, normalized, and augmented for training.
   ➤ Dataset Source: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

2. **🔍 Model Architecture**

   * Pre-trained **VGG16** model used as a feature extractor
   * Final classifier layers customized for binary classification
   * Built and trained using **PyTorch**

3. **📊 Training & Validation**

   * Binary Cross-Entropy Loss
   * Metrics tracked: Accuracy, Precision, Recall
   * Training on GPU-supported environment (if available)

4. **🌐 Streamlit Web App**
   Users can upload MRI images and receive instant predictions.
   ➤ **Live App**: [Brain Tumor Detection App](https://brain-tumor-detection-system-using-pytorch.streamlit.app/)

5. **📦 Model Hosting on Hugging Face**
   The trained model is uploaded to Hugging Face for sharing and reuse.
   ➤ **Model Link**: [brain-tumor-detection-model](https://huggingface.co/mtalhazafar/brain-tumor-detection-model/)

---

## 📁 Repository Structure

```
Brain-Tumor-Detection/
├── app.py                              # Streamlit web app
├── Brain_Tumor_Detection_VGG16.ipynb   # Model training & analysis notebook
├── requirements.txt                    # List of dependencies
├── README.md                           # Project overview and instructions
```

---

## Getting Started

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**

```bash
git clone https://github.com/MTalhaZafar32/Brain-Tumor-Detection-Using-PyTorch-VGG16-Transfer-Learning/.git
cd Brain-Tumor-Detection
```

2. **Create a Virtual Environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

3. **Install Required Packages**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**

```bash
streamlit run app.py
```

---

## 📂 Data Files

* **Dataset**: Brain MRI images categorized into `yes` (tumor) and `no` (normal) folders
  ➤ [Download from Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

* **Trained Model**:
  ➤ [View on Hugging Face](https://huggingface.co/your-username/brain-tumor-detection-model)

---

## 🧪 How to Use

1. Run the app using the Streamlit command above.
2. Upload an MRI image via the web interface.
3. The model will classify the image as **Tumor** or **Normal** in real-time.

Or try it directly here:
🌐 [Streamlit Live App](https://brain-tumor-detection-system.streamlit.app/)

---

## 🤝 Contributions

Have suggestions, ideas, or improvements? Fork the repo, make your changes, and open a pull request. All contributions are welcome!

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
