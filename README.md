# Machine Learning API Service

A FastAPI-based web service that trains machine learning models from Excel data and makes predictions. Runs in Docker containers with a modern web interface.

## Features

- Upload Excel files and train ML models
- Multiple algorithms: Logistic Regression, SVM, Random Forest, Neural Networks
- RESTful API with Swagger documentation
- Modern web interface
- Docker containerization

## Quick Start

```bash
# Clone and run
git clone <repository-url>
cd project_case
docker compose up --build

# Open in browser
http://localhost:8000
```

## Usage

1. **Upload Data**: Upload Excel file (.xlsx) through web interface
2. **Configure Model**: Select algorithm and hyperparameters
3. **Train Model**: Start training and view performance metrics
4. **Make Predictions**: Use trained model for predictions

## API Endpoints

- `POST /upload` - Upload Excel file
- `POST /config` - Set algorithm and parameters
- `POST /train` - Train model
- `POST /predict` - Make prediction
- `GET /metrics` - View model performance

## Project Structure

```
app/
├── main.py          # FastAPI app
├── models/          # Data models
├── routes/          # API endpoints
├── services/        # Business logic
├── static/          # Web interface
└── utils/           # Utilities
```

## Tech Stack

- FastAPI
- Scikit-learn
- Pandas
- Docker
- HTML/CSS/JavaScript


