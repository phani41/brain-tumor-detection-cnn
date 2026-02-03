# 🧠 Brain Tumor Detection Frontend

React + Vite frontend for brain tumor classification with model comparison interface. Allows users to upload MRI images and compare predictions from MobileNet vs EfficientNet models.

## 🚀 Features

- **Model Comparison**: Side-by-side MobileNet vs EfficientNet results
- **Best Model Badge**: Highlights the model with higher confidence
- **Interactive Charts**: Probability visualization with Chart.js/Recharts
- **Image Upload**: Drag & drop or browse functionality
- **Invalid Image Detection**: Warns when uploaded image is not a valid MRI
- **Responsive Design**: Clean and minimal UI

## 🖥️ Tech Stack

- **React 19** - Frontend framework
- **Vite** - Build tool and dev server
- **Chart.js** - Data visualization
- **Recharts** - React chart library
- **Axios** - HTTP client
- **CSS3** - Styling

## 📂 Project Structure

```
frontend/
├── brain tumor/                    # Main React app
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadImage.jsx     # File upload component
│   │   │   ├── ResultCard.jsx     # Model result display
│   │   │   ├── BestModelBadge.jsx # Best model indicator
│   │   │   ├── ProbabilityChart.jsx # Individual probability chart
│   │   │   └── ComparisonCharts.jsx # Side-by-side charts
│   │   ├── pages/
│   │   │   └── ComparePage.jsx     # Main comparison page
│   │   ├── services/
│   │   │   └── api.js              # Backend API calls
│   │   ├── App.jsx                 # Main app component
│   │   └── main.jsx                # App entry point
│   ├── package.json                # Dependencies
│   └── vite.config.js              # Vite configuration
└── README.md                       # This file
```

## 🔧 Installation

1. **Navigate to React app directory**
```bash
cd brain-tumor-cnn/frontend/brain\ tumor/
```

2. **Install dependencies**
```bash
npm install
```

## ▶️ Development

```bash
npm run dev
```

Frontend runs at: **http://localhost:5173**

## 🏗️ Build

```bash
npm run build
```

## 📱 Components

### UploadImage.jsx
- File upload with drag & drop
- Image preview
- File validation

### ResultCard.jsx  
- Model prediction display
- Confidence percentage
- Probability breakdown

### BestModelBadge.jsx
- Highlights superior model
- Dynamic styling based on performance

### ProbabilityChart.jsx
- Individual model probability visualization
- Dark charts on light UI theme

### ComparisonCharts.jsx
- Side-by-side model comparison
- Interactive probability charts

## 🔗 API Integration

Backend endpoint: `http://127.0.0.1:5000/compare`

**Request:**
```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await axios.post('/compare', formData);
```

**Response Structure:**
```javascript
{
  invalid: false,
  best_model: "efficientnet",
  mobilenet: {
    prediction: "glioma",
    confidence: 87.45,
    probabilities: {...}
  },
  efficientnet: {
    prediction: "glioma", 
    confidence: 92.18,
    probabilities: {...}
  }
}
```

## 🎨 UI Features

- **Light Theme**: Clean white background
- **Dark Charts**: High contrast data visualization  
- **Responsive Layout**: Works on desktop and mobile
- **Error Handling**: User-friendly error messages
- **Loading States**: Visual feedback during processing

## 🏥 Tumor Classes

- **Glioma** - Malignant brain tumor
- **Meningioma** - Usually benign tumor  
- **No Tumor** - Healthy brain tissue
- **Pituitary** - Pituitary gland tumor

## 📋 Dependencies

```json
{
  "react": "^19.2.0",
  "vite": "^7.2.4", 
  "axios": "^1.13.4",
  "chart.js": "^4.5.1",
  "react-chartjs-2": "^5.3.1",
  "recharts": "^3.7.0"
}
```

## ⚠️ Important Notes

- **Educational Use Only** - Not for medical diagnosis
- **Backend Required** - Ensure Flask API is running on port 5000
- **Valid Images Only** - Upload MRI brain scans for accurate results
- **Model Comparison** - Results show both MobileNet and EfficientNet predictions

## 🚀 Getting Started

1. Start the backend server (port 5000)
2. Install frontend dependencies
3. Run `npm run dev`
4. Open http://localhost:5173
5. Upload an MRI image to compare models