# 🧠 Brain Tumor Detection Backend

Flask API backend for brain tumor classification using CNN models. Supports model comparison between MobileNet and EfficientNet architectures.

## 🚀 Features

- **Dual Model Support**: MobileNet vs EfficientNet comparison
- **Image Validation**: Detects invalid/non-MRI images
- **CORS Enabled**: Cross-origin requests supported
- **Health Check**: API status endpoint
- **Confidence Thresholding**: Invalid image detection below 45% confidence

## 🖥️ Tech Stack

- **Flask** - Web framework
- **TensorFlow** - Deep learning models
- **PIL** - Image processing
- **NumPy** - Numerical computations
- **Flask-CORS** - Cross-origin support

## 📂 Project Structure

```
backend/
├── model/
│   ├── brain_tumor_model.keras          # MobileNet model
│   └── brain_tumor_efficientnet.keras   # EfficientNet model
├── app.py                               # Main Flask application
├── requirements.txt                     # Python dependencies
├── train_model.py                       # MobileNet training script
└── train_model_efficientnet.py         # EfficientNet training script
```

## 🏥 Model Classes

- **glioma** - Malignant brain tumor
- **meningioma** - Usually benign tumor
- **notumor** - No tumor detected
- **pituitary** - Pituitary gland tumor

## 🔧 Installation

1. **Clone repository**
```bash
git clone <repository-url>
cd brain-tumor-cnn/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure models exist**
```
backend/model/brain_tumor_model.keras
backend/model/brain_tumor_efficientnet.keras
```

## ▶️ Run Server

```bash
python app.py
```

Server runs at: **http://127.0.0.1:5000**

## 📡 API Endpoints

### Health Check
```http
GET /
```
**Response:**
```json
{"status": "Brain Tumor API running"}
```

### Model Comparison
```http
POST /compare
Content-Type: multipart/form-data
```

**Request:**
- `image`: MRI image file (JPG/PNG)

**Response (Valid Image):**
```json
{
  "invalid": false,
  "best_model": "efficientnet",
  "mobilenet": {
    "prediction": "glioma",
    "confidence": 87.45,
    "probabilities": {
      "glioma": 87.45,
      "meningioma": 8.32,
      "notumor": 2.15,
      "pituitary": 2.08
    }
  },
  "efficientnet": {
    "prediction": "glioma", 
    "confidence": 92.18,
    "probabilities": {
      "glioma": 92.18,
      "meningioma": 4.21,
      "notumor": 2.01,
      "pituitary": 1.60
    }
  }
}
```

**Response (Invalid Image):**
```json
{
  "invalid": true,
  "message": "Uploaded image is not a valid brain MRI"
}
```

## 🔍 Image Processing

- **Input Size**: 224x224 pixels
- **MobileNet**: Standard normalization (0-1)
- **EfficientNet**: EfficientNet preprocessing
- **Invalid Threshold**: <45% confidence = invalid image

## ⚠️ Important Notes

- **Educational Use Only** - Not for medical diagnosis
- **Model Files Required** - Ensure both .keras files exist
- **Image Validation** - Only accepts valid image formats
- **CORS Enabled** - Allows frontend connections

## 🚀 Training Models

Run training scripts to create new models:

```bash
# Train MobileNet model
python train_model.py

# Train EfficientNet model  
python train_model_efficientnet.py
```