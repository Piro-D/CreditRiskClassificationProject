# 5th Semester Data Mining Project

Group Members : 
- Jason Nicholas Rahardjo   - 2802411411
- Bryan Vincent	            - 2802557232
- Yohannes Adrian		    - 2802404721
- Georgius Brandon		    - 2802407465
- Duard Elbert 			    - 2820405642



# KDD Process

## 1. Data selection
The dataset that will be used to train the various models will be sourced from the South German Credit dataset, which is a dataset from UC Irvine machine learning repository licensed under Creative Commons, that corrected mistranslations in the infamous Statlog German credit dataset from 1994 [3]. The dataset contains 20 predictor variables and 1 target variable, with 1000 rows of data. The predictor variables involve characteristics of the applicant that indicate a person’s capability to pay back the proposed loan/credit, such as loan amount, employment duration, installment rate, residence, property, savings, etc. Although the dataset does not contain any missing values, preprocessing is still required to appropriately encode the categorical predictors, improve the interpretability, and prepare it for the classification model.

## 2. Data Preprocessing
For preprocessing the following steps will be conducted

## 3. Exploratory Data Analysis
Exploratory data analysis (EDA) was conducted to understand the structure of the data and to decide how it should be prepared for modelling. It covered the class balance, the distribution of the numerical predictors, and the relationship between each predictor and the target variable (credit risk).

The dataset contains 1,000 applicants, of which 700 (70%) are good credit risks and 300 (30%) are bad credit risks. This class imbalance means that accuracy alone is a misleading metric, so the models will also be compared using metrics such as recall, precision, and F1-score for the bad class, and the data will be split in a stratified way.

The numerical predictors (loan duration, loan amount, and age) are all right-skewed, and outliers were identified using the 1.5 x IQR rule but not removed, since large loans are valid observations. Duration and amount are moderately correlated (0.62), while age is almost unrelated to both. The relationship between these predictors and credit risk is weak when measured by correlation (duration -0.21, amount -0.15, age 0.09), but grouping them shows clearer patterns. For example, the bad rate rises from 21% for loans of 12 months or less to 52% for loans of 37 months or more.

Among the categorical predictors, credit history, checking account status, savings, and other debtors show the largest differences in bad rate across categories. For instance, applicants with past payment delays have a bad rate of 62.5%, compared with 17% for applicants whose credits at the bank were all paid back duly. In contrast, predictors such as telephone, number of people liable, and present residence show very small differences.

These findings motivate the transformation step: the categorical predictors will be encoded, skewed numerical predictors such as loan amount will be transformed, and very small categories will be merged or regularised. Some category labels (for example property and purpose) look inconsistent with the observed bad rates and should be verified against the dataset documentation.
