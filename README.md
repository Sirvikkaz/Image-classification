# Project Title
A simple web application that classifies images into 10 categories using a CNN model trained on the CIFAR-10 dataset.

## Categories
Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Features
* Convolutional Neural Network for image classification (e.g., CIFAR-10)
* FastAPI backend for model inference
* Streamlit frontend for interactive demos

## Performance
- **Accuracy**: 82.9%
- **Note**: Images outside these categories will result in inaccurate predictions

## Tech Stack
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML**: TensorFlow/Keras, OpenCV, numpy
- **Model**: CNN trained on CIFAR-10

## Project Structure
```
├── backend/            # FastAPI server code and model files
├── frontend/           # Streamlit app code
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

## Installation
1. Clone the repository:
    ```bash
   git clone https://github.com/Sirvikkaz/Image-classification.git
   cd Image-classification
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### FastAPI Backend
1. Navigate to the backend folder:
    ```bash
   cd backend
   ```
2. Start the server:
    ```bash
   fastapi run dev main.py --host 127.0.0.1 --port 9090
   ```

### Streamlit Frontend
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Run Streamlit:
   ```bash
   streamlit run streamlit.py
   ```

## Webapp usage
1. Open the Streamlit app in your browser
2. Wait for "Model ready!" message
3. Upload an image (JPG, PNG, JPEG)
4. Click "Get Prediction"
5. View the classification result


## Limitations
- Only works with the 10 CIFAR-10 categories, larger dataset maybe used in the future
- Model will misclassify images outside the training categories

## License
MIT