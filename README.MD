# 5th Semester Data Mining Project

Group Members : 
- Jason Nicholas Rahardjo   - 2802411411
- Bryan Vincent	            - 2802557232
- Yohannes Adrian		    - 2802404721
- Georgius Brandon		    - 2802407465
- Duard Elbert 			    - 2820405642

# KDD Process

The project follows the KDD process to prepare the South German Credit dataset and compare classification models for credit risk. Data selection, data preprocessing, exploratory data analysis, and data transformation have been completed in the current project. Model training and evaluation will be conducted in the following stages.

## 1. Data Selection

The dataset that will be used to train the various models is the South German Credit dataset from the UC Irvine Machine Learning Repository. It provides corrected coding information for the widely used Statlog German credit dataset from 1994. The dataset contains 20 predictor variables and 1 target variable, with 1,000 rows of data. The predictor variables involve characteristics of the applicant that indicate a person's capability to pay back the proposed loan/credit, such as loan amount, employment duration, installment rate, residence, property, and savings.

The target variable, `credit_risk`, indicates whether an applicant is a good or bad credit risk. The raw data is stored in `Dataset/raw/SouthGermanCredit.asc`, and the meaning of the categorical codes is provided in `Dataset/raw/codetable.txt`. Although the dataset does not contain any missing values, preprocessing is still required to check the data quality, improve the interpretability, and prepare it for the classification models.

## 2. Data Preprocessing

Data preprocessing was conducted in (notebooks/datapreprocessing.ipynb) to inspect the raw data and make the column names and categorical values easier to understand. The categorical codes were decoded using the provided code table, while the numerical predictors were kept in their original supplied values. Encoding and scaling for modelling are handled separately in the data transformation stage.

The tasks completed in data preprocessing were:

- Load the whitespace-delimited raw dataset and inspect its shape, columns, data types, and sample records.
- Rename the original German column names to their corresponding English names.
- Read the code table and create mappings between categorical codes and their meanings, including the target variable.
- Classify the predictors into 3 numerical, 11 nominal, and 6 ordinal features.
- Check for missing values. No missing values were found, so imputation was not required.
- Check for exact duplicate records. No duplicate records were found, so no rows were removed.
- Compare the observed categorical codes with the documented codes and identify any invalid or unused codes.
- Check the numerical predictors for their minimum and maximum values, non-finite values, and non-positive values.
- Inspect the target class distribution.
- Replace the categorical and target codes with readable labels, preserving the documented order of the ordinal categories in memory.
- Save the result to `Dataset/processed/preprocessed.csv` without a DataFrame index column.

The preprocessed dataset retains all 1,000 observations and 21 columns. It contains readable category labels and has not yet been scaled or encoded into model inputs.

## 3. Exploratory Data Analysis

Exploratory data analysis (EDA) was conducted in (notebooks/eda.ipynb) to understand the structure of the data and to decide how it should be prepared for modelling. It covered the class balance, the distribution of the numerical predictors, and the relationship between each predictor and the target variable (credit risk).

The dataset contains 1,000 applicants, of which 700 (70%) are good credit risks and 300 (30%) are bad credit risks. This class imbalance means that accuracy alone is a misleading metric, so the models will also be compared using metrics such as recall, precision, and F1-score for the bad class, and the data is split in a stratified way.

The numerical predictors (loan duration, loan amount, and age) are all right-skewed, and outliers were identified using the 1.5 x IQR rule but not removed, since large values can be valid observations. Duration and amount are moderately correlated (0.62), while age is almost unrelated to both. The relationship between these predictors and credit risk is weak when measured by correlation (duration -0.21, amount -0.15, age 0.09), but grouping them shows clearer patterns. For example, the bad rate rises from 21% for loans of 12 months or less to 52% for loans of 37 months or more.

Among the categorical predictors, credit history, checking account status, savings, and other debtors show the largest differences in bad rate across categories. For instance, applicants with past payment delays have a bad rate of 62.5%, compared with 17% for applicants whose credits at the bank were all paid back duly. In contrast, predictors such as telephone, number of people liable, and present residence show very small differences. Ordinal features were also compared using their documented category order and Spearman correlation with the target.

These findings motivated the encoding and numerical transformations in the next stage.

## 4. Data Transformation

Data transformation was conducted in (notebooks/data_transformation.ipynb) to prepare the train and test datasets and define reusable preprocessing components. The notebook demonstrates how the transformations affect the training data, while the exported features remain in their original numerical and categorical representations, with the additional engineered feature. This allows preprocessing to be fitted separately within each training fold during model validation.

The tasks completed in data transformation were:

- Encode the target variable as `bad = 0` and `good = 1`.
- Create `amount_per_month` by dividing amount by duration, after checking that duration is finite and greater than zero. 
- Split the data into 70% training and 30% testing using stratification and `random_state=42`.
- Compare the original, training, and testing class counts and proportions to confirm stratification.
- Analyse training-data skewness and outliers before and after transformation, using the correct IQR fences of Q1 - 1.5 x IQR and Q3 + 1.5 x IQR.
- Define log1p transformation, winsorization at the 1st and 99th percentiles, and StandardScaler for the continuous features.
- Define OrdinalEncoder using the existing category orders, followed by StandardScaler, for the ordinal features.
- Define OneHotEncoder with `handle_unknown="ignore"` for nominal features.
- Demonstrate the transformations using temporary fitted copies on training rows only, and inspect the resulting means, population standard deviations, and missing-value counts.
- Expose a fresh, unfitted preprocessor through `build_preprocessor()` in [src/preprocessing_pipeline.py](src/preprocessing_pipeline.py).
- Export four datasets without index columns, keeping the feature and target rows aligned and excluding the target from the feature datasets.

The exported datasets are stored in `Dataset/processed/`:

| Dataset | Observations | Contents |
| --- | ---: | --- |
| `X_train.csv` | 700 | 21 features before fitted preprocessing |
| `y_train.csv` | 700 | 490 good and 210 bad target labels |
| `X_test.csv` | 300 | The same 21 feature columns in the same order |
| `y_test.csv` | 300 | 210 good and 90 bad target labels |



