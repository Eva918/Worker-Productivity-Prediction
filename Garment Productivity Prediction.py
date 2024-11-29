#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
get_ipython().system('pip install xgboost')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ## Data Preparation
# 

# In[2]:


df = pd.read_csv("C:/Users/Evang/Downloads/garments_worker_productivity.csv")
df


# In[3]:


print(df.dtypes)


# In[4]:


df.info()


# In[5]:


# Check for missing values
missing_values = df.isnull().sum()

# Display the count of missing values
print("Missing values in each column:")
print(missing_values)


# In[6]:


df['wip'] = df['wip'].fillna(df['wip'].median())

# Check if the missing values have been replaced
print(df.isnull().sum())


# In[7]:


# Set figure size
plt.figure(figsize=(12, 8))

# Bar chart for 'quarter'
plt.subplot(2, 2, 1)
sns.countplot(x='quarter', data=df, palette=["#d7e1ee", "#cbd6e4", "#bfcbdb", "#b3bfd1", "#a4a2a8", "#df8879", "#c86558", "#b04238", "#991f17"])
plt.title('Distribution of Quarter')
plt.xlabel('Quarter')
plt.ylabel('Count')

# Bar chart for 'department'
plt.subplot(2, 2, 2)
sns.countplot(x='department', data=df, palette=["#d7e1ee", "#cbd6e4", "#bfcbdb", "#b3bfd1", "#a4a2a8", "#df8879", "#c86558", "#b04238", "#991f17"])
plt.title('Distribution of Department')
plt.xlabel('Department')
plt.ylabel('Count')

# Bar chart for 'day'
plt.subplot(2, 2, 3)
sns.countplot(x='day', data=df, palette=["#d7e1ee", "#cbd6e4", "#bfcbdb", "#b3bfd1", "#a4a2a8", "#df8879", "#c86558", "#b04238", "#991f17"])
plt.title('Distribution of Day')
plt.xlabel('Day')
plt.ylabel('Count')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# ### Exploratory Data Analysis (EDA)

# In[8]:


# Scatter plot for targeted vs actual productivity
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='targeted_productivity', y='actual_productivity')
plt.title('Targeted Productivity vs Actual Productivity')
plt.xlabel('Targeted Productivity')
plt.ylabel('Actual Productivity')
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set inline plotting for Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the distribution of the 'actual_productivity' column
plt.figure(figsize=(8, 4))
sns.histplot(df['actual_productivity'], kde=True, color='plum', edgecolor="k", linewidth=1)
plt.title('Distribution of Actual Productivity')
plt.xlabel('Actual Productivity')
plt.ylabel('Frequency')
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set inline plotting for Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the distribution of the 'actual_productivity' column
plt.figure(figsize=(8, 4))
sns.histplot(df['targeted_productivity'], kde=True, color='plum', edgecolor="k", linewidth=1)
plt.title('Distribution of Targeted Productivity')
plt.xlabel('Targeted Productivity')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# save correlations to variable
corr = df.corr()

#create a mask to not show duplicate values
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# generate heatmap
plt.figure(figsize= (12,12))
sns.heatmap(corr, annot=True, center=0, mask=mask, cmap='gnuplot')
plt.show()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatterplot to visualize relationships between actual_productivity and other variables
plt.figure(figsize=(12, 8))

# Scatterplot: actual_productivity vs overtime
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='over_time', y='actual_productivity', color='plum')
plt.title('Actual Productivity vs Overtime')
plt.xlabel('Overtime')
plt.ylabel('Actual Productivity')

# Scatterplot: actual_productivity vs incentive
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='incentive', y='actual_productivity', color='teal')
plt.title('Actual Productivity vs Incentive')
plt.xlabel('Incentive')
plt.ylabel('Actual Productivity')

# Scatterplot: actual_productivity vs smv
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='smv', y='actual_productivity', color='orange')
plt.title('Actual Productivity vs SMV')
plt.xlabel('SMV')
plt.ylabel('Actual Productivity')

plt.tight_layout()
plt.show()



# ### Data Preprocessing

# In[13]:


# Boxplot to visualize distributions and potential outliers for each variable
plt.figure(figsize=(12, 8))

# Boxplot: actual_productivity
plt.subplot(2, 2, 1)
sns.boxplot(data=df, y='actual_productivity', color='plum')
plt.title('Distribution of Actual Productivity')

# Boxplot: overtime
plt.subplot(2, 2, 2)
sns.boxplot(data=df, y='over_time', color='teal')
plt.title('Distribution of Overtime')

# Boxplot: incentive
plt.subplot(2, 2, 3)
sns.boxplot(data=df, y='incentive', color='orange')
plt.title('Distribution of Incentive')

# Boxplot: smv
plt.subplot(2, 2, 4)
sns.boxplot(data=df, y='smv', color='skyblue')
plt.title('Distribution of SMV')

plt.tight_layout()
plt.show()


# In[14]:


# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[['actual_productivity', 'over_time', 'incentive', 'smv']].quantile(0.25)
Q3 = df[['actual_productivity', 'over_time', 'incentive', 'smv']].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers by filtering values that fall within the bounds
df_clean = df[~((df[['actual_productivity', 'over_time', 'incentive', 'smv']] < lower_bound) | (df[['actual_productivity', 'over_time', 'incentive', 'smv']] > upper_bound)).any(axis=1)]

# Verify if outliers were removed
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")


# In[15]:


# You can now replot the boxplots after cleaning the data
plt.figure(figsize=(12, 8))

# Boxplot: actual_productivity
plt.subplot(2, 2, 1)
sns.boxplot(data=df_clean, y='actual_productivity', color='plum')
plt.title('Distribution of Actual Productivity (Outliers Removed)')

# Boxplot: overtime
plt.subplot(2, 2, 2)
sns.boxplot(data=df_clean, y='over_time', color='teal')
plt.title('Distribution of Overtime (Outliers Removed)')

# Boxplot: incentive
plt.subplot(2, 2, 3)
sns.boxplot(data=df_clean, y='incentive', color='orange')
plt.title('Distribution of Incentive (Outliers Removed)')

# Boxplot: smv
plt.subplot(2, 2, 4)
sns.boxplot(data=df_clean, y='smv', color='skyblue')
plt.title('Distribution of SMV (Outliers Removed)')

plt.tight_layout()
plt.show()


# ### Machine Learning Model and Evaluation

# In[16]:


print(df['quarter'].unique())
print(df['department'].unique())
df['department'] = df['department'].str.strip()


# In[17]:


# Convert 'day' column to boolean
df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Saturday':5, 'Sunday': 6})
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['quarter'] = df['quarter'].map({'Quarter1': 1, 'Quarter2': 2, 'Quarter3': 3, 'Quarter4': 4, 'Quarter5':5})
df['department'] = df['department'].map({'sweing': 1, 'finishing': 2})


# In[18]:


#Data Splitting
X = df.drop(columns=['actual_productivity', 'date'])  # Features
y = df['actual_productivity']  # Target variable


# In[19]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


from sklearn.preprocessing import StandardScaler
# Select columns for scaling (continuous numerical columns only)
columns_to_scale = X_train.select_dtypes(include=[np.number]).columns.difference(['day', 'quarter', 'department'])
scaler = StandardScaler()

# Apply StandardScaler only on continuous numerical columns
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])


# In[21]:


# For pandas DataFrames or Series, use the .dtypes attribute
print("Data types of X_train:")
print(X_train.dtypes)

print("\nData types of X_test:")
print(X_test.dtypes)

print("\nData types of y_train:")
print(y_train.dtypes)

print("\nData types of y_test:")
print(y_test.dtypes)


# In[31]:


from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1)  # Adjusted to ensure valid range
}


# In[32]:


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
model = xgb.XGBRegressor(objective="reg:squarederror")

# Initialize GridSearchCV with cross-validation
grid_search =RandomizedSearchCV(
    estimator=XGBRegressor(),
    param_distributions=param_dist_xgb,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_iter=50,  # Number of random configurations to try
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)



# In[24]:


y_pred = best_model.predict(X_test)


# In[25]:


# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# R-squared (R²)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


# In[26]:


# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(best_model, importance_type='weight')
plt.title('Feature Importance')
plt.show()


# In[27]:


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Define the range of training set sizes to evaluate
train_sizes = np.linspace(0.1, 1.0, 10)

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model,       # Use the best model from grid search
    X=X_train,                  # Training features
    y=y_train,                  # Training target
    train_sizes=train_sizes,    # Sizes of the training set to evaluate
    cv=5,                       # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Calculate mean and standard deviation of train and test scores
train_scores_mean = -np.mean(train_scores, axis=1)  # Convert negative scores to positive MSE
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)    # Convert negative scores to positive MSE
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Error', color='blue', marker='o')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='blue', alpha=0.1)

plt.plot(train_sizes, test_scores_mean, label='Validation Error', color='orange', marker='o')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='orange', alpha=0.1)

# Adding titles and labels
plt.title("Learning Curve for XGBoost Model")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Squared Error")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[28]:


#cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import numpy as np



# In[29]:


# Define the scoring functions
scoring = {
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'R2': make_scorer(r2_score)
}

# Perform cross-validation with 5-folds for each metric
mae_scores = cross_val_score(best_model, X, y, cv=5, scoring=scoring['MAE'])
mse_scores = cross_val_score(best_model, X, y, cv=5, scoring=scoring['MSE'])
r2_scores = cross_val_score(best_model, X, y, cv=5, scoring=scoring['R2'])

# Convert MSE scores to RMSE by taking the square root
rmse_scores = np.sqrt(-mse_scores)


# In[35]:


print(f"Mean Absolute Error (MAE): {-np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Mean Squared Error (MSE): {-np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"R-squared (R²): {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")


# In[ ]:




