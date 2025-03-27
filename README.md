# Music Recommendation System using Big Data

A scalable and efficient music recommendation system built with PySpark and modern recommendation algorithms.

## Project Structure

```
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/           # Recommendation models
│   │   ├── popularity.py
│   │   ├── collaborative_filtering.py
│   │   └── hybrid.py
│   └── evaluation/       # Evaluation metrics and utilities
│       ├── metrics.py
│       └── validation.py
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
├── config/              # Configuration files
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Features

- **Multiple Recommendation Approaches**:
  - Popularity-based models
  - Collaborative Filtering (ALS)
  - Hybrid models with time-based weighting
  - Advanced ranking systems

- **Data Processing**:
  - Efficient data preprocessing pipeline
  - Feature engineering
  - Data validation and cleaning

- **Evaluation**:
  - MAP (Mean Average Precision)
  - NDCG (Normalized Discounted Cumulative Gain)
  - Comprehensive model comparison

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your environment:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

## Usage

1. Data Preprocessing:
```bash
python src/data/preprocessing.py
```

2. Train Models:
```bash
python src/models/train.py
```

3. Evaluate Models:
```bash
python src/evaluation/evaluate.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
(Open Source MIT License)
