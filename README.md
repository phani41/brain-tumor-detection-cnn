# Brain Tumor Detection using CNN

A full-stack medical AI application for brain tumor classification from MRI images using dual model comparison.

## Features
- **Dual Model Comparison**: MobileNetV2 vs EfficientNet predictions
- **Smart Model Selection**: Automatic best model decision based on confidence scores
- **4-Class Classification**: Glioma, Meningioma, Pituitary, No Tumor
- **REST API Backend**: Flask with CORS support
- **React Frontend**: Real-time image analysis with interactive charts
- **Visualization**: Probability charts and confidence comparisons
- **Uncertainty Handling**: Low confidence detection and warnings

## Tech Stack
- **Backend**: Python, TensorFlow, Flask, OpenCV, PIL
- **Frontend**: React, Vite, Chart.js, Recharts
- **Models**: MobileNetV2, EfficientNet (both Keras format)

## Project Structure
```
brain-tumor-cnn/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # Model training script
│   ├── requirements.txt    # Python dependencies
│   └── model/
│       ├── brain_tumor_model.keras
│       └── brain_tumor_efficientnet.keras
├── frontend/
│   └── brain tumor/
│       ├── src/
│       │   ├── App.jsx              # Main app component
│       │   ├── pages/
│       │   │   └── ComparePage.jsx  # Comparison page
│       │   ├── components/
│       │   │   ├── UploadImage.jsx
│       │   │   ├── ResultCard.jsx
│       │   │   ├── BestModelBadge.jsx
│       │   │   ├── ComparisonCharts.jsx
│       │   │   └── ProbabilityChart.jsx
│       │   └── services/
│       │       └── api.js           # Backend API calls
│       └── vite.config.js
└── README.md
```

## API Endpoints

### GET `/`
Health check endpoint
```json
{ "status": "Brain Tumor API running" }
```

### POST `/predict`
Single model prediction (MobileNetV2)
```json
{
  "model": "MobileNetV2",
  "prediction": "glioma",
  "confidence": 95.5,
  "probabilities": {
    "glioma": 95.5,
    "meningioma": 2.1,
    "notumor": 1.2,
    "pituitary": 1.2
  }
}
```

### POST `/compare`
Compare both models
```json
{
  "mobilenet": {
    "prediction": "glioma",
    "confidence": 95.5,
    "probabilities": { ... }
  },
  "efficientnet": {
    "prediction": "glioma",
    "confidence": 92.3,
    "probabilities": { ... },
    "warning": "OK"
  },
  "best_model": {
    "model": "MobileNetV2",
    "prediction": "glioma"
  }
}
```

## How to Run

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API will run on `http://127.0.0.1:5000`

### Frontend Setup
```bash
cd "frontend/brain tumor"
npm install
npm run dev
```

The frontend will run on `http://localhost:5173` (or configured Vite port)

## Models

Both models are trained on the Brain Tumor MRI Dataset with:
- **Input Size**: 224×224 RGB images
- **Classes**: 4 (glioma, meningioma, notumor, pituitary)
- **Training Data Augmentation**: Rotation, zoom, shifts, horizontal flips
- **Training Strategy**: 
  - Phase 1: 10 epochs with frozen base (learning rate: 1e-3)
  - Phase 2: 5 epochs with fine-tuning (learning rate: 1e-5)

## Decision Logic

The `/compare` endpoint selects the best model based on:
1. If MobileNetV2 confidence ≥ EfficientNet confidence AND ≥ 50% → **MobileNetV2**
2. Else if EfficientNet confidence ≥ 50% → **EfficientNet**
3. Else → **Uncertain** (low confidence warning)

## Image Requirements
- Format: JPG, PNG, or other standard image formats
- Resolution: Any (will be resized to 224×224)
- Type: MRI brain scan images for best results

## Notes
- Confidence scores below 40% trigger low confidence warnings
- Preprocessing varies per model (MobileNetV2 vs EfficientNet normalization)
- API runs with `debug=True` by default (change for production)