# Career Recommendation System

A machine learning-based career recommendation system that helps students choose their career paths based on their academic performance and interests.

## Project Structure

```
Career-Recommandation/
├── backend/           # Backend implementation
│   ├── src/          # Source code files
│   │   ├── ML.py     # Main machine learning implementation
│   │   ├── ML2.py    # Additional ML functionality
│   │   └── main.py   # Application entry point
│   ├── data/         # Data files
│   │   └── student.csv  # Student dataset
│   └── README.md     # Backend documentation
├── frontend/         # Frontend implementation
│   ├── src/         # Source code files
│   ├── public/      # Static files
│   └── README.md    # Frontend documentation
├── requirements.txt  # Python dependencies
├── .gitignore       # Git ignore rules
├── LICENSE          # MIT License
└── README.md        # This file
```

## Features

- Career path recommendations based on academic performance
- User-friendly web interface
- Machine learning-based prediction system
- Personalized career suggestions
- Real-time API integration

## Prerequisites

- Python 3.x
- Node.js (for frontend)
- Required Python packages (see requirements.txt)
- Required Node.js packages (see frontend/package.json)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chetakk/Career-Recommandation.git
cd Career-Recommandation
```

2. Set up the backend:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
python src/main.py
```

3. Set up the frontend:
```bash
# Install Node.js dependencies
cd frontend
npm install

# Start the frontend development server
npm start
```

4. Open your browser and navigate to `http://localhost:3000`

## Development

### Backend
- Built with Python
- Uses scikit-learn for ML models
- Flask for API endpoints
- Pandas for data processing

### Frontend
- Built with React.js
- Material-UI for components
- Axios for API calls
- Responsive design

## Environment Variables

### Backend
Create a `.env` file in the backend directory:
```
FLASK_APP=src/main.py
FLASK_ENV=development
```

### Frontend
Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=http://localhost:5000
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for the tools and libraries used in this project