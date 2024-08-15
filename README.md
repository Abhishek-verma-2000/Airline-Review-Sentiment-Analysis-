# Airline-Review-Sentiment-Analysis-

Owner : Abhishek Verma - 23M0031

Airline Review Sentiment Analysis

Project Overview:
This machine learning project aimed to predict customer sentiment (positive/negative recommendation) based on airline reviews. The project utilized natural language processing and classification techniques to analyze text reviews and predict overall customer sentiment.

Key Technologies and Libraries:
- Python
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for machine learning algorithms and preprocessing
- NLTK and spaCy for natural language processing
- Matplotlib for data visualization (implied but not directly shown in code)

Data Processing and Feature Engineering:
- Dataset: 23,171 airline reviews
- Text preprocessing: Lowercase conversion, punctuation removal, stopword removal, lemmatization
- Feature extraction: TF-IDF vectorization for review text and titles
- Additional features: Numerical ratings (e.g., seat comfort, staff service)
- Final feature matrix: 38,048 features

Machine Learning Model:
- Algorithm: Logistic Regression
- Train-test split: 80% train, 20% test (stratified sampling)
- Training set size: 18,536 samples
- Test set size: 4,635 samples

Model Performance:
- Accuracy: 95.40%
- Precision: 93.21%
- Recall: 93.15%
- F1-Score: 93.18%

Confusion Matrix:
- True Negatives: 2,967
- False Positives: 106
- False Negatives: 107
- True Positives: 1,455

Class-wise Performance:
- Negative class (not recommended):
  - Precision: 0.97
  - Recall: 0.97
  - F1-score: 0.97
  - Support: 3,073 samples

- Positive class (recommended):
  - Precision: 0.93
  - Recall: 0.93
  - F1-score: 0.93
  - Support: 1,562 samples

Key Achievements:
1. Successful implementation of a text classification pipeline, combining NLP techniques with machine learning.
2. High accuracy (95.40%) in predicting customer recommendations based on review text and numerical ratings.
3. Balanced performance across both positive and negative classes, indicating robust model generalization.
4. Efficient handling of high-dimensional data (38,048 features) using sparse matrices and appropriate algorithms.

This project demonstrates proficiency in data preprocessing, feature engineering, text analysis, and machine learning model development and evaluation. The high accuracy and balanced performance across classes showcase the effectiveness of the implemented approach in sentiment analysis tasks.
