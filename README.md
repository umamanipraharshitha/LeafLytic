# 🌿 LeafLytic

**LeafLytic** is an intelligent deep learning-powered web application for detecting plant leaf diseases. With a fine-tuned **MobileNetV2** model, LeafLytic identifies **15 different types of leaf diseases** with an impressive **92% accuracy**. The tool aims to assist farmers, gardeners, and agricultural experts with fast and reliable crop health diagnosis.
- Contributors: Ameerunnisa Khan {https://github.com/Ameer-design351} , Venkata Omanand{https://github.com/Venkataomanand}
---

## 🚀 Features

- 📷 Upload leaf images for real-time disease detection
- 🧠 Deep learning model (MobileNetV2) – 92% accuracy
- 🔬 Classifies 15 different disease types
- 📝 Shows disease name, confidence score, and treatment tips
- 🌐 Clean, intuitive frontend using HTML, CSS, JavaScript
- 🖥️ Backend with Flask (Python) serving the ML model

---

## 🧠 Deep Learning Model

- **Model:** MobileNetV2 (transfer learning, fine-tuned)
- **Classes:** 15 leaf diseases (e.g., Apple Scab, Corn Rust, Healthy, etc.)
- **Accuracy:** 92% on test data
- **Framework:** TensorFlow/Keras
- **Input Size:** 224x224 RGB images
- **Training Techniques:**
  - Image augmentation (rotation, flipping, zoom)
  - Optimizer: Adam
  - Loss: Categorical Crossentropy

---

## 🔧 Tech Stack

| Layer         | Technology         |
|---------------|--------------------|
| Frontend      | HTML, CSS, JavaScript |
| Backend       | Flask (Python)     |
| Deep Learning | TensorFlow, Keras  |
| Model         | MobileNetV2        |

---


---

## 💻 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/umamanipraharshitha/LeafLytic.git
cd LeafLytic
2. Create a virtual environment and install dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate     # For Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Run the Flask App
bash
Copy
Edit
python app.py
Visit: http://127.0.0.1:5000
