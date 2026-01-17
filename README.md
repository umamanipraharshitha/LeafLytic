

# Leaf Disease Detection Web App

A web application to identify plant leaf diseases using a **pretrained EfficientNetB0** model. Users can log in with Google, upload leaf images, and get predictions with confidence scores. The model achieves **~92% accuracy** on the test dataset.

---

## Features

* **Google OAuth Login** for secure access.
* **Leaf Disease Detection** for 16 classes (Tomato, Potato, Pepper; healthy & diseased).
* **Upload Images** and get prediction with confidence.
* **Data Augmentation** during model training.
* **Responsive Web Pages:** Home, LeafLens, Search, About, Contact, Results.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/leaf-disease-detector.git
cd leaf-disease-detector
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up Google OAuth:

   * Place `client_secret.json` in project root.
   * Redirect URI: `http://127.0.0.1:5000/callback`.

5. Ensure upload folder exists:

```bash
mkdir -p static/uploads
```

---

## Usage

1. Run the app:

```bash
python app.py
```

2. Open in browser: `http://127.0.0.1:5000/`

3. Login with Google to access protected pages (LeafLens, Search).

4. Upload a leaf image on **LeafLens** to get prediction and confidence.

---

## Model Details

* **Architecture:** EfficientNetB0 (pretrained on ImageNet)
* **Input Size:** 224 x 224 RGB
* **Output Classes:** 16 leaf conditions
* **Accuracy:** ~92%
* **Techniques:** Data augmentation, class weighting, fine-tuning top layers, then deeper layers

---

## File Structure

```
leaf-disease-detector/
├── app.py
├── leaf_disease_detector.h5
├── client_secret.json
├── dataset/          # optional for retraining
├── static/uploads/   # uploaded images
├── templates/        # HTML pages
└── requirements.txt
```

---

## Dependencies

* Flask
* TensorFlow / Keras
* Pillow
* NumPy
* scikit-learn
* google-auth, google-auth-oauthlib
* werkzeug

Co-authored-by: Ameerunnisa Khan <ameerunnisakhan786@gmail.com>
Co-authored-by: Venkata Omanand <venkataomanand@gmail.com>"
