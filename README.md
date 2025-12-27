Toxic Comment Classification with Ensemble Models

Overview:

This project tackles multi-label toxic comment classification using the Kaggle Wikipedia Talk Page Toxicity dataset. Each comment may belong to zero, one, or multiple toxicity categories:
toxic, severe_toxic, obscene, threat, insult, identity_hate. Our goal was to explore multiple modeling strategies for text classification, compare their performance, and ultimately combine them into an ensemble model to improve robustness and generalization

Dataset:
- Source: Kaggle – Wikipedia Talk Page Toxic Comments
Labels:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

Key Challenges:
Severe class imbalance, with labels like threat and identity_hate appearing far less frequently than toxic. Wide variation in comment length, including very long toxic comments

Data Preprocessing:
We applied the following preprocessing steps:
- Filtered dataset to keep only comment text and toxicity labels
- Removed missing text entries
- Converted all text to lowercase strings
- Vectorized text using TF-IDF
- Removed stop words
- Filtered rare terms using min_df = 5
- Addressed class imbalance using class_weight="balanced" in applicable models

Models Implemented:
Each team member explored a different modeling approach:
1. TF-IDF + Logistic Regression (One-vs-Rest)
Word-level TF-IDF with n-grams
One logistic regression classifier per label
Strong baseline performance for high-frequency labels
Effective for high-dimensional sparse text data
 178 Project Writeup
2. TF-IDF + Support Vector Classifier
Word-level TF-IDF without n-grams
Support Vector Classifier wrapped with CalibratedClassifierCV to enable probability outputs
Lower memory usage and faster training
Slightly less expressive due to loss of positional information
 178 Project Writeup
3. Character-Level TF-IDF + Multinomial Naive Bayes
Character n-grams (4–6 characters)
Designed to capture misspellings and obfuscated toxic language
One-vs-Rest classification setup
Particularly useful for detecting intentionally altered offensive words
 178 Project Writeup

Ensemble Approach:
Given that all individual models achieved similar overall performance, we implemented an ensemble model to leverage their complementary strengths.

Initial ensemble:
- Used stacking with a Support Vector Classifier
- Tuned regularization parameter C, with C = 1.0 performing best
- Precision-Recall curves showed strong performance on common labels (toxic, obscene, insult)
- Performance dropped on rarer labels due to limited training examples

Neural Network Experiments:
Following TA feedback, we experimented with a Multi-Layer Perceptron (MLP):
- Observed overfitting (low training error, higher validation error)
- Implemented early stopping to mitigate overfitting
- identity_hate failed to train properly due to extreme class imbalance
- Tried different optimizers (e.g., Adam vs SGD), but results did not surpass the SVM-based ensemble

Results & Insights
All baseline models achieved >90% overall accuracy, making marginal gains difficult. Improvements beyond ~1% required extensive tuning and architectural changes
Ensemble modeling helped reduce overfitting and handle edge cases. Class imbalance remained the dominant limiting factor, especially for rare labels

The project highlighted the distinction between good and great models in real-world NLP tasks

Technologies Used:
- Python
- scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Support Vector Machines
- Naive Bayes
- Multi-Layer Perceptrons
- Precision-Recall evaluation metrics

Future Work:
- Apply boosting methods (e.g., AdaBoost, Gradient Boosting)
- Explore transformer-based models (BERT-style architectures)
- Improve handling of extreme class imbalance
- Further optimize ensemble weighting strategies

Contributors:
- Kaelyn
- Preston
- Thomas
