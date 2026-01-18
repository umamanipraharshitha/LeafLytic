
# 🌿 Leaf Disease Detection Web Application (Flask + MobileNetV2)

## 1️⃣ Objective

This project is a **web-based plant disease detection application** that allows users to:

* Upload an image of a leaf
* Automatically classify it into one of **15 plant disease categories** (or healthy)
* Get a **confidence score** for the prediction
* Access the app securely via **Google OAuth login**

The system is designed for **farmers, researchers, and students** to quickly and accurately detect plant diseases.

---

## 2️⃣ Dataset & Model

* **Dataset:** PlantVillage (15 classes, including healthy leaves)
* **Model:** MobileNetV2 (pre-trained on ImageNet, fine-tuned on PlantVillage)
* **Input size:** 224×224×3
* **Training details:**

  * Initial training with frozen base layers
  * Fine-tuning last 40 layers for improved accuracy
  * Data augmentation: rotation, zoom, shift, horizontal flip
* **Performance:**

  * Test Accuracy: ~93.8%
  * Macro F1-score: 0.93
  * Weighted F1-score: 0.94
* **Model file:** `leaf_disease_model.keras`

---

## 3️⃣ Web Application Features

### 🔹 Authentication

* Uses **Google OAuth 2.0** for secure login
* Supports profile fetching: name, email, profile picture
* Protected pages with `login_required` decorator

### 🔹 Pages

**Public:**

* Home
* About
* Contact

**Protected (login required):**

* LeafLens (upload & prediction)
* Search
* CompleteSignup

### 🔹 Leaf Disease Prediction

* Users upload leaf images via `/leaflens` page
* Uploaded images are:

  * Preprocessed to 224×224 RGB format
  * Normalized to `[0,1]`
  * Sent through the **MobileNetV2 model**
* Predictions returned:

  * Disease class label
  * Confidence score (%)
  * Display of uploaded leaf image

### 🔹 File Handling

* Uploaded images saved in `static/uploads`
* Secure filenames handled with `werkzeug.utils.secure_filename`
* Predictions rendered on `results.html` with confidence and image

---

## 4️⃣ Technical Stack

* **Backend:** Flask (Python)
* **Authentication:** Google OAuth 2.0
* **Deep Learning:** TensorFlow + Keras
* **Image Processing:** PIL, NumPy
* **Frontend:** HTML templates (home.html, leaflens.html, results.html, etc.)
* **Deployment:** Local development via `app.run(debug=True)`; can be deployed on cloud platforms

---

## 5️⃣ Key Advantages

* **Lightweight model:** MobileNetV2 (~3.9M params) suitable for web or mobile deployment
* **High accuracy:** 93–94% across 15 classes
* **Secure:** Google OAuth ensures only authenticated users can access protected pages
* **User-friendly:** Upload, view results, and confidence visualization in one workflow

---

## 6️⃣ Potential Enhancements

1. **Explainable AI:** Integrate Grad-CAM to show which parts of the leaf influenced the prediction
2. **Mobile Deployment:** Convert `.keras` model to `.tflite` for Android apps
3. **Additional Diseases:** Expand dataset to more plant species and diseases
4. **Real-time Detection:** Integrate live camera feed prediction

---

### 7️⃣ Conclusion

This **Flask-based Leaf Disease Detection Web App** combines **state-of-the-art deep learning** (MobileNetV2) with a **secure, user-friendly web interface**. It achieves **high accuracy**, provides **reliable predictions**, and is ready for deployment for agricultural decision support.




Co-authored-by: Ameerunnisa Khan <ameerunnisakhan786@gmail.com>
Co-authored-by: Venkata Omanand <venkataomanand@gmail.com>"
