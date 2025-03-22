# Backend

This directory contains the backend implementation of the Career Recommendation System.

## Directory Structure

```
backend/
├── src/           # Source code files
│   ├── ML.py     # Main machine learning implementation
│   ├── ML2.py    # Additional ML functionality
│   └── main.py   # Application entry point
├── data/         # Data files
│   └── student.csv  # Student dataset
└── README.md     # This file
```

## Setup

1. Install dependencies:

```bash
pip install -r ../requirements.txt
```

2. Run the application:

```bash
python src/main.py
```

## API Endpoints

The backend provides the following API endpoints:

- `/api/recommend` - Get career recommendations based on student data
- `/api/health` - Health check endpoint

## Data

The `data/` directory contains the training and testing datasets used by the ML models.
