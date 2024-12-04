# Predicting Surgical Site Infections with Machine Learning and Deep Learning

This repository provides a comprehensive pipeline for predicting surgical site infections (SSI) using machine learning (ML) and deep learning (DL) techniques.

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/etiennechlt/ssi-predict.git
cd ssi-predict
```

### 2. Install Dependencies
Ensure you have Python 3.7+ installed. Install the required packages:
```
pip install -r requirements.txt
```

### 3. Configure the Pipeline
Edit the config.json file to specify:
- Feature restrictions.
- Excluded procedures and age filters.

### 4. Run the Training Pipeline
```
python src/training_pipeline.py
```

### 5. Explore Results
- Metrics: Check the results/metrics.csv.
- Saved Models: Stored in the models/ folder for reuse.
