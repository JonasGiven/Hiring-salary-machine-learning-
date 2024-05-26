
### 1. Data Preprocessing

The preprocessing steps are crucial for ensuring your data is clean and ready for modeling. Here’s a quick summary and some additional tips:

#### Reading the Data
```python
import pandas as pd

# Load the dataset
df_hiring = pd.read_csv('/content/hiring 2024-05-24 05_26_29.csv')
df_hiring.head()
```

#### Handling Missing Values
- Fill missing values in the `experience` column with `0`.
- Fill missing values in `test_score(out of 10)` with the median.

```python
# Fill NaN values in experience with 0
df_hiring['experience'].fillna(0, inplace=True)

# Fill NaN values in test_score(out of 10) with median
median_test_score = df_hiring['test_score(out of 10)'].median()
df_hiring['test_score(out of 10)'].fillna(median_test_score, inplace=True)
```

#### Converting Text to Numeric
- Convert text values in the `experience` column to numeric values.

```python
# Mapping experience text to numeric values
experience_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                      'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11}

df_hiring['experience'] = df_hiring['experience'].replace(experience_mapping).astype(int)
```

### 2. Model Training

Next, you train the linear regression model:

```python
from sklearn.linear_model import LinearRegression

# Create the linear regression model
reg = LinearRegression()

# Train the model
reg.fit(df_hiring[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],
        df_hiring['salary($)'])
```

### 3. Model Coefficients and Intercept

Extracting the model parameters can help in understanding the model’s behavior:

```python
# Model coefficients
coefficients = reg.coef_
print("Coefficients:", coefficients)

# Model intercept
intercept = reg.intercept_
print("Intercept:", intercept)
```

### 4. Predictions

Make predictions using the trained model:

```python
# Predict the salary for a candidate with 14 years experience, 9 test score, and 7 interview score
predicted_salary = reg.predict([[14, 9, 7]])
print("Predicted Salary:", predicted_salary)
```

### Additional Enhancements

1. **Model Evaluation**:
   - Use metrics like Mean Squared Error (MSE) or R-squared to evaluate the model’s performance on the training data.
   ```python
   from sklearn.metrics import mean_squared_error, r2_score

   # Make predictions on the training set
   predictions = reg.predict(df_hiring[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']])

   # Calculate MSE
   mse = mean_squared_error(df_hiring['salary($)'], predictions)
   print("Mean Squared Error:", mse)

   # Calculate R-squared
   r2 = r2_score(df_hiring['salary($)'], predictions)
   print("R-squared:", r2)
   ```

2. **Data Visualization**:
   - Visualize the relationships between features and the target variable to gain insights.
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   sns.pairplot(df_hiring)
   plt.show()
   ```

3. **Cross-Validation**:
   - Use cross-validation to ensure the model's robustness and avoid overfitting.
   ```python
   from sklearn.model_selection import cross_val_score

   # Perform cross-validation
   cv_scores = cross_val_score(reg, df_hiring[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],
                               df_hiring['salary($)'], cv=5)

   print("Cross-validation scores:", cv_scores)
   print("Average cross-validation score:", cv_scores.mean())
   ```
