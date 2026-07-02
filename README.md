# 🦷 Teeth Classification  

## 📌 Project Overview  
This project focuses on **classifying dental images into 7 distinct categories** using deep learning.  
It explores two different approaches:  
1. **From Scratch:** Custom CNN inspired by VGG16.  
2. **Transfer Learning:** MobileNetV2 with fine-tuning.  

Additionally, the pretrained model is deployed using **Streamlit** to provide an interactive web application.  

---

## ⚙️ Approaches  

### 🔹 Approach 1 – From Scratch (Simplified VGG16)  
- **Data Preprocessing:**  
  - Used `ImageDataGenerator` with rescaling and augmentations (rotation, zoom, shifts, flips).  
  - Dataset split into train (80%), validation (10%), and test (10%).  
- **Model Architecture:**  
  - 3 × Conv2D + MaxPooling layers.  
  - Dropout layers for regularization.  
  - Dense + Softmax output.  
- **Callbacks:** EarlyStopping & ModelCheckpoint.  
- **Performance:**  
  - Validation Accuracy: ~92%  
  - Test Accuracy: ~97.3%  
  - Test Loss: ~0.082  
- **Saved Model:** `best_teeth_model.h5`  

---

### 🔹 Approach 2 – Transfer Learning (MobileNetV2 + Fine-Tuning)  
- **Base Model:** MobileNetV2 pretrained on ImageNet.  
- **Custom Head:** GlobalAveragePooling + Dense layers.  
- **Training Strategy:**  
  - Initially froze most of MobileNetV2.  
  - Fine-tuned top layers for better adaptation.  
- **Performance:**  
  - Test Accuracy: **99.12%** 🎯  
  - Test Loss: **~0.07**  
- **Saved Model:** `best_teeth_PretrainedModel.h5`  

---

### 🚀 Deployment (Streamlit Web App)  
A web interface was developed using **Streamlit** to allow users to upload dental images and receive predictions instantly.  

Run locally with:  
```bash
streamlit run app.py
```
## 📊 Performance Comparison  

| Approach         | Model             | Test Accuracy | Test Loss |
|------------------|------------------|---------------|-----------|
| From Scratch     | Simplified VGG16 | ~97.3%        | ~0.082    |
| Transfer Learning| MobileNetV2      | **99.12%**    | ~0.07     |

## 📂 Project Structure  

- 📁 **notebooks**  
  - 📓 `teeth_classification_from_scratch.ipynb` – CNN from scratch  
  - 📓 `teeth_classification_PretrainedModel.ipynb` – MobileNetV2 + fine-tuning  
- 📄 `app.py` – Streamlit web app  
- 📄 `Task.pdf` – Project description  
- 📄 `README.md` – Documentation  



