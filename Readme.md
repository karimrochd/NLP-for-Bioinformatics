# NLP for BioInformatics

## Students
- Karim Rochd
- Celina Bedjou


## Table of Contents
1. [Introduction](#introduction)
2. [A baseline model with bag of words](#a-baseline-model-with-bag-of-words)
   2.1 [Overview](#overview)
   2.2 [Data Acquisition and preparation](#data-acquisition-and-preparation)
   2.3 [Feature Selection via PCA for Enhanced TF-IDF Vectorization](#feature-selection-via-pca-for-enhanced-tf-idf-vectorization)
   2.4 [Comparative Evaluation of Machine Learning Models](#comparative-evaluation-of-machine-learning-models)
   2.5 [Observations](#observations)
   2.6 [Results and Discussion](#results-and-discussion)
3. [Pretrained Biomedical Embeddings Model](#pretrained-biomedical-embeddings-model)
   3.1 [Overview of BioSentVec Model](#overview-of-biosentvec-model)
   3.2 [Data Preprocessing and Preparation](#data-preprocessing-and-preparation)
   3.3 [Initial Model Training and Selection](#initial-model-training-and-selection)
   3.4 [Validation and Comparative Results](#validation-and-comparative-results)
   3.5 [Selection and Scaling Up](#selection-and-scaling-up)
   3.6 [The Impact of BioSentVec Pre-trained Embeddings](#the-impact-of-biosentvec-pre-trained-embeddings)
4. [Integration of Deep Learning](#integration-of-deep-learning)
   4.1 [The Added Value of Deep Learning in Biomedical Text Classification](#the-added-value-of-deep-learning-in-biomedical-text-classification)
5. [Conclusion](#conclusion)

## Introduction
This project addresses the challenge of sentence classification in biomedical research using the PubMed 200k RCT dataset. We explore three distinct strategies for sentence classification, each employing different methodologies to capture the semantic and contextual nuances in the data.

## A baseline model with bag of words

### Overview
Our project focuses on classifying scientific texts from the PubMed 200k RCT dataset, which contains sentences from abstracts categorized into Background, Objective, Methods, Results, and Conclusions. We implement various machine learning models including Support Vector Machines (SVM), Logistic Regression, and Random Forest classifiers, combined with advanced text preprocessing and feature extraction techniques.

### Data Acquisition and preparation
We sourced the dataset from a GitHub repository and unpacked it from a compressed 7z file. The labels were converted to integer format to streamline the model training process.

### Feature Selection via PCA for Enhanced TF-IDF Vectorization
We used Principal Component Analysis (PCA) to identify and select the most influential features based on their variance contributions, optimizing the TF-IDF vectorization process.

### Comparative Evaluation of Machine Learning Models
We conducted a detailed comparative analysis of the following models:
- Support Vector Machine (SVM) with a Linear Kernel
- Logistic Regression
- Random Forest Classifier

### Observations
Performance metrics such as precision, recall, F-score, and confusion matrices were used to evaluate each model's effectiveness. The SVM model demonstrated superior performance.

### Results and Discussion
The models showed commendable performance in categorizing text, with the SVM model slightly outperforming others. The use of PCA for feature selection and Random OverSampling to mitigate class imbalance proved beneficial.

## Pretrained Biomedical Embeddings Model

### Overview of BioSentVec Model
We employed the BioSentVec model, a sophisticated sentence embedding tool that uses a 700-dimensional space based on the sent2vec framework and a bigram approach.

### Data Preprocessing and Preparation
We utilized the PubMed_200k_RCT dataset, employing SciSpacy for tasks such as lemmatization and abbreviation expansion. Multiprocessing techniques were used to reduce processing time.

### Initial Model Training and Selection
We initially trained three models (SVM, Logistic Regression, and Random Forest) on a smaller subset of the data to identify the most effective classifier.

### Validation and Comparative Results
Results for each model:
- SVM: Precision 0.647, Recall 0.659, F-score 0.651
- Logistic Regression: Precision 0.678, Recall 0.674, F-score 0.674
- Random Forest: Precision 0.534, Recall 0.549, F-score 0.493

### Selection and Scaling Up
Logistic Regression emerged as the most promising classifier and was trained on a larger dataset of 50,000 sentences, showing improved performance.

### The Impact of BioSentVec Pre-trained Embeddings
BioSentVec significantly enhanced the classifiers' discernment capabilities by providing rich semantic embeddings that capture intricate relationships in biomedical texts.

## Integration of Deep Learning

We incorporated a deep learning model into our methodology, achieving a precision of 0.7546 on the test corpus.

### The Added Value of Deep Learning in Biomedical Text Classification
The integration of deep learning introduced a new dimension of complexity and adaptiveness, excelling in identifying and learning from subtle patterns within the data.

## Conclusion

Our comparative analysis of traditional machine learning models and a deep learning approach revealed distinct advantages offered by each methodology. The deep learning model demonstrated superior performance, particularly in accuracy and pattern recognition. The synergy created by combining deep learning with BioSentVec embeddings and traditional classifiers showcases a potent approach for handling the nuances of biomedical text classification. Our findings advocate for a hybrid approach, leveraging the strengths of both traditional and modern machine learning techniques to address the challenges of biomedical text classification.