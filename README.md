# Statistical Model for Predicting Above Average Academic Performance

## Introduction

The purpose of this project is to build a statistic model based on a binary logistic regression algorithm for predicting above average academic performance by students across three different metrics: math, reading, and writing. Results of the binary classification algorithm will be used to determine which major factors contribute/ is associated with above average exam performance based on estimations of the logistic regression parameters (coefficients) of each factor. Significance of each factor in predicting above average academic performance of students will be evaluated at a confidence level of 95% with a 0.05 significance threshold for each of the three logistic model in measuring above average performance. This analysis was completed in `Python Jupyter Notebook` application using Anaconda Distribution.

<br>

## Requirements for Execution

The file is to be executed using a Python Notebook application such as Jupyter Notebook via Anaconda, or Google Colab Notebook.

#### Dataset 
Dataset: Student Performance in Exams

Filename: `exams.csv`

The project requires the import of a stimulated dataset of  `Student Performance in Exams` and has been provided in a `csv` format listed as `exams.csv` (Chauhan, 2022). Alternatively, as this dataset was extracted from an online data repository platform, this file can be accessed and downloaded directly from Kaggle, an online data repository. Users can use this [link](https://www.kaggle.com/datasets/whenamancodes/students-performance-in-exams) to be redirected to the webpage (note: will need to register for an account).

The dataset consists of 1000 students and 8 features including the following:
- **Gender** - Male, Female (categorical)
- **Race/Ethnicity** - Student race/ethnicity classified based on groups (A-E) (categorical)
- **Parental Level of Education** - Some high school, high school, some college, associate's degree, bachelor, master's (categorical)
- **Lunch** - Standard, Free/Reduced (categorical)
- **Test Preparation Course** - Whether the student has completed a course in preparation for the examination (categorical)
- **Math, Reading, Writing Scores** - marks secured by students in each of the subject (quantitative)

#### Importing the required packages 

 Support from `Anaconda Distribution` featuring a compilation of open-source data science and machine learning packages was used in this analysis.

In order to run the `ipynb` file locally, the following modules are required:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
```

If not already installed - modules can be downloaded using the `pip` command followed by the package name:

```python
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install statsmodels
```

Data visualization will be supported by the use of the `matplotlib` and `seaborn` library as part of the exploratory data analysis phase (EDA).

In order to display figures created, the following command is to be executed at the start of the program. This will embed static images of the plot in the output of new cells in the notebook:

```python
%matplotlib inline
```

#### Reading the file

To run the `ipynb` file, the dataset should be located in the same directory as code itself and imported into a`dataframe` object as a `.csv` format as provided. This is to be executed using `pd.read_csv()` function by passing a string argument of the file name including its extension:

```python
pd.read_csv("exams.csv")
```

If the file is not found in the same directory, pass in the file path as the argument.

<br>

## Features/Outputs of the Analysis

The analysis of `Student Performance in Exams` is divided into two main components:
1. Exploratory Data Analysis (EDA)
2. Fitting the logistic regression model to the data

#### Part 1: Exploratory Data Analysis (EDA)

**Data Cleaning and Processing**
- As part of the EDA and data cleaning process, the program will rename column names to yield more appropriate syntax for further data manipulation and output the adjusted dataset to be used for the remainder of the analysis.

- The program will also display the following information about the dataset including the dimensionality/size of the dataset, the different data types involved, and presence of any potential missing values that need to be addressed. 

**Descriptive Statistics** 
- EDA will also feature summary statistics of the overall exam scores secured by students exam for each of the three subjects in math, reading, and writing. 

- At a more granular level, the analysis will also output the average exam score for each subject compared across different classes within each of the 5 factors (gender, race/ethnicity, parental level of education, lunch plan, and whether the student took a test preparation course).

**Data Visualizations**

- Mainly box plots and bar charts were produced in the program to visualize the spread and distribution of the dataset:

	1. Bar charts in displaying the distribution of students grouped in class intervals for each category. A single bar chart will be displayed upon execution for each of the 5 factors.
	
	2. Box plots in visualizing the overall shape of the distribution of each examination and by factor grouped by class intervals.

- Additionally, a correlation heat map of exam scores is also displayed to illustrate how a student's performance in one subject is related to the performance in another subject

#### Part 2: Fitting the logistic regression to the data

**Testing for multicollinearity**
- As the statistical model will involve more than two independent variables, the program will output a correlation heat map of to identify any covariates that are highly correlated among predictors in the dataset. 

- This is accomplished by recoding each class in a category to numeric values and based on the results of the correlation values, high covariates were to be excluded if present.

**Regression model**
- Regression model was completed using  the `statsmodel` library.

- Three separate logistic model are fitted in this analysis for each subject in determining which factors contribute/associated with above average student performance in math, reading and writing as the target variable in each of the separate models.

- To predict above average student performance, marks secured by each of the student in the dataset were recoded as 0 - for below average performance, or 1 - for above average performance in the dataset. This is based on whether the score is above or below the overall average in math, reading, and writing.

- Dummy variables are created for each class in a category in order to run the regression.

- During the fitting of the model, the dataset is separated to a feature matrix containing the predictors, and a target array contain the column of the outcome variable (math, reading, or writing). 

- Summary of the Regression results is outputted for each model once fitted.

**Interpreting the regression results**

- The logistic regression model is used to predict the outcome of a binary outcome  (0 - below average, 1 - above average. Therefore, for each of the regression model, it can be used to estimate the probability of event 1 occurring based on predictors (i.e., above average student performance in math, reading or writing)

- This means that every unit increase in the predictor variable is associated with an increase or decrease (depending on the sign of the estimated logistic coefficient for the given variable) of the log odds of the the student performing above average in math, reading, or writing, by the value of the coefficient. 

- Note that when interpreting coefficient values for dummy variables, odds of above average performance is relative to the reference class (class that is omitted from report). For example, for the logistic regression model for math (logit_math), the predicted log odds of females performing above average in math is 0.87 lower than males; this was found to be statistically significant from 0 at a 95% confidence level.

- Results are considered statistically signifiant from 0 if the corresponding p-value ($P>|z|$) is less than 0.05.

- The constant is interpreted as the predicted log odds of a student performing above average in math, reading or writing, when all predictors are at reference level (these are the classes for each of the categorical variable/factors that were omitted).
<br>

## Running the Program

1. Open a Python Notebook application (e.g., Jupyter Notebook or Google Colab)

2. Load code file and ensure that dataset is found in the same directory.

3. Load all necessary modules and packages indicated in the "Requirements for Execution" section (install if necessary)

4. Import "Student Performance in Exams" dataset as a csv file: `exams.csv` using the pandas module.

5. Once requirements are satisfied, users can run the entire program at once by using the `Run All` command to execute all of the cells at once. This will produce all the output at once. Alternatively, users can choose to execute each cell individually.

<br>

## References

Dataset:

```
Chauhan, A. (2022, August). Students Performance in Exams. Retrieved October 23, 2022 from https://www.kaggle.com/datasets/whenamancodes/students-performance-in-exams. 
```
