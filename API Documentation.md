# API Documentation

### 1. Creating a bar graph - `bar_graph()`

```python
def bar_graph(cat):
	feature = student_performance[cat]
	
	yvals = feature.value_counts()
	xvals = feature.value_counts().index

	plt.ylabel("frequency")
	plt.title(cat)
	
	return plt.bar(xvals, yvals)
```

**Description:**
- `bar_graph()` takes a pandas Series object of a dataframe object as an argument and outputs a bar graph displaying the frequency (number of students) for each class intervals in the category passed in the function.

- To plot, the function takes the corresponding counts of each unique value in the Series object for each class interval as the y-values using `value_counts()` and the corresponding index of the returned object as the x-labels.

**Parameters:**
- `cat` - pandas series object corresponding to the categorical variable in the dataset in which a bar chart is to be displayed of the frequency of each class interval.

<br>

### 2. Creating a box plot - `box_plot()`

```python
def box_plot(var, score, plot_index):
	df = student_performance
	plot = sns.boxplot(data=df, x=var, y=score, ax=axes[plot_index])
	plot.set(xlabel=None)
	return plot
```

**Description:**
- `box_plot()` outputs a box-plot containing the distribution of each class interval for a given categorical variable as single subplot to a figure object with n row by m columns that has been defined outside of the function. In this analysis, a 1 by 3 figure object is defined, such that each figure will contain 3 independent subplots for math, reading, and writing.

- This will display the distribution of scores for each examination for each class interval of a categorical variable. 

**Parameters:**
- `var` - categorical variable containing class intervals and passed as a pandas Series object.

- `score` - pandas Series object of the exam variable to visualize the distribution of scores for each different classes of a categorical variable.

- `plot_index` - specifies the index position in which the subplot is to be positioned in figure object.

<br>


### 3. Creating binary outcomes for classifications

To predict above average student performance in math, reading, and writing, each examination score were recoded as 0 - for below average performance, or 1 - for above average performance in the dataset. 

To recode each student mark, the following functions below were passed as arguments to the `pd.Dataframe.apply()`method accepts any user defined function to transformation elements in a dataframe object. Transformations were applied column-wise on the Student Performance in Exams dataframe object for each respective exam. 

**1. Creating binary outcomes for above/below average performance in math**

```python
def above_below_math(score):
	avg = summary.loc["mean"]["math_score"]
	if score > avg:
		return 1 # above
	else:
		return 0 # below
```

**Description:**
- This function is used perform element-wise transformations with Pandas' `apply()` to recode examinations scores in math by comparing the value of the current element to the overall average math score of the dataset.

- The function will return the value of 1 if the score is found to be greater the average math score of the dataset, and 0 otherwise.

**Parameters:**
- `score` - mark secured by the student in math to be recoded 1 or 0 relative to the average math score of the dataset

<br>


**2. Creating binary outcomes for above/below average performance in reading** 

```python
def above_below_reading(score):
	avg = summary.loc["mean"]["reading_score"]
	if score > avg:
		return 1 # above
	else:
		return 0 # below
```

**Description:**
- This function is used perform element-wise transformations with Pandas' `apply()` to recode examinations scores in reading by comparing the value of the current element to the overall average reading score of the dataset.

- The function will return the value of 1 if the score is found to be greater the average reading score of the dataset, and 0 otherwise.

**Parameters:**
- `score` - mark secured by the student in reading to be recoded 1 or 0 relative to the average reading score of the dataset

<br>

**3. Creating binary outcomes for above/below performance in writing**

```python
def above_below_writing(score):
	avg = summary.loc["mean"]["writing_score"]
	if score > avg:
		return 1 # above
	else:
		return 0 # below
```

**Description**
- This function is used perform element-wise transformations with Pandas' `apply()` to recode examinations scores in writing by comparing the value of the current element to the overall average writing score of the dataset.

- The function will return the value of 1 if the score is found to be greater the average writing score of the dataset, and 0 otherwise.

**Parameters**
- `score` - mark secured by the student in writing to be recoded 1 or 0 relative to the average writing score of the dataset


<br>

### 4. Fitting the regression model

```python
def fit_logit(score:str):
	# defining feature matrix for the model
	feature_matrix = new_df[['gender_female',
	'race/ethnicity_group A', 'race/ethnicity_group B',
	'race/ethnicity_group D', 'race/ethnicity_group E',
	"parental_level_of_education_associate's degree",
	"parental_level_of_education_bachelor's degree",
	'parental_level_of_education_high school',
	"parental_level_of_education_master's degree",
	'parental_level_of_education_some high school', 'lunch_free/reduced',
	'test_preparation_course_completed']]
	
	X_label = feature_matrix
	# adding intercept to the model
	X_label = sm.add_constant(X_label)
	# defining model target/label
	y = new_df[score]
	
	logit_model=sm.Logit(y, X_label) # creating instance of the logistic regression model
	result=logit_model.fit() # fitting model to the data

	return result
```

**Description:**
- For each invocation, the function creates an instance of a logistic model from the class `statsmodels.discrete.discrete_model.Logit ` and fits the model to the data in determination of which factors contribute/is associated with above average performance exams.

- The function divides the dataset into, (1) a two-dimensional feature matrix of predictors that has been converted into dummy variables (X_label), and (2) a single Series object of the target variable (y) specified in the argument, both of which are passed as arguments to the `sm.Logit()` function to create an instance of the logistic model.

- The `fit()`  function is called on the object to "fit" the model to the data

- Note that because the `statsmodel` module does not take the intercept into account for the regression. To compensate for this, the function includes an additional line of code to include a column of ones to the matrix of features using the following function:

```python
X_label = sm.add_constant(X_label)
```

**Parameters:**
- `score` - string argument of the exam performance you want to predict as the outcome variable of the model.