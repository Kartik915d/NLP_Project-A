# Trend Detection in Fashion Project

## Course: NLP (Semester 6) - Pillai College of Engineering
## Project Overview:

This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The project focuses on Trend Detection in Fashion, where we apply various Machine Learning (ML), Deep Learning (DL), and Language Models to categorize news articles into predefined categories. This project involves exploring techniques like text preprocessing, feature extraction, model training, and evaluating the models for their effectiveness in classifying news articles.

You can learn more about the college by visiting the official website of Pillai College of Engineering.

## Acknowledgements:
We would like to express our sincere gratitude to the following individuals:

Theory Faculty:<br>
Dhiraj Amin<br>
Sharvari Govilkar<br>

Lab Faculty:<br>
Dhiraj Amin<br>
Neha Ashok<br>
Shubhangi Chavan

Their guidance and support have been invaluable throughout this project.<br>

## Project Title:
Trend Detection in Fashion using Natural Language Processing

## Project Abstract:
This project develops an NLP-based classification system to categorize fashion products into trend-related groups—seasonal, vintage, and trending—using product descriptions. A custom dataset is created and processed with NLP techniques like tokenization, stopword removal, and TF-IDF or word embeddings. Machine learning models, including Naive Bayes, SVM, and Random Forest, are trained to optimize classification accuracy. By integrating NLP-driven trend analysis, retailers can improve inventory management, align with consumer preferences, and enhance profitability in a dynamic fashion market.

### Algorithms Used:
### Machine Learning Algorithms:<br>
- Logistic Regression<br>
- Support Vector Machine (SVM)<br>
- Random Forest Classifier<br>

### Deep Learning Algorithms:<br>
- Convolutional Neural Networks (CNN)<br>
- Recurrent Neural Networks (RNN)<br>
- Long Short-Term Memory (LSTM)<br>

### Language Models:<br>
- GPT<br>
- BERT (Bidirectional Encoder Representations from Transformers)<br>
## Comparative Analysis:
The comparative analysis of different models highlights their effectiveness in classifying news articles into the correct category. The following table summarizes the accuracy, precision, recall, and F1-score of the models tested:
### Machine Learning Models

### Deep Learning Models
| No. | Model Name  | Feature  | Precision | Recall | F1 Score | Accuracy |
|----|------------|---------|-----------|--------|----------|----------|
| 1  | CNN       | BoW     | 0.16      | 0.23   | 0.17     | 0.225    |
| 2  | LSTM      | BoW     | 0.05      | 0.23   | 0.08     | 0.225    |
| 3  | BiLSTM    | BoW     | 0.05      | 0.23   | 0.08     | 0.225    |
| 4  | CNN-BiLSTM | BoW    | 0.05      | 0.23   | 0.08     | 0.225    |
| 5  | CNN       | TF-IDF  | 0.05      | 0.23   | 0.08     | 0.225    |
| 6  | LSTM      | TF-IDF  | 0.05      | 0.23   | 0.08     | 0.225    |
| 7  | BiLSTM    | TF-IDF  | 0.05      | 0.23   | 0.08     | 0.225    |
| 8  | CNN-BiLSTM | TF-IDF | 0.05      | 0.23   | 0.08     | 0.225    |
| 9  | CNN       | FastText| 0.05      | 0.23   | 0.08     | 0.225    |
| 10 | LSTM      | FastText| 0.05      | 0.23   | 0.08     | 0.225    |
| 11 | BiLSTM    | FastText| 0.05      | 0.23   | 0.08     | 0.225    |
| 12 | CNN-BiLSTM | FastText| 0.05     | 0.23   | 0.08     | 0.225    |


### Language Models
| # | Model Name | Precision | Recall | Accuracy | F1 Score |
|---|------------|-----------|--------|----------|----------|
| 1 | RoBERTa | 0.0529 | 0.2300 | 0.2300 | 0.0860 |
| 2 | BERT | 0.0529 | 0.2300 | 0.2300 | 0.0860 |

## Conclusion:
This Trend detection in fashion project demonstrates the potential of Machine Learning, Deep Learning, and Language Models for text classification tasks, particularly for categorizing news articles. The comparative analysis reveals that BERT, a transformer-based model, outperforms traditional methods and deep learning models in terms of accuracy, precision, and recall. By employing various algorithms, we gain insights into the strengths and weaknesses of each model, allowing us to choose the most suitable approach for news classification.
