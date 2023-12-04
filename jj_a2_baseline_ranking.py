import pandas as pd
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

### Don't forget to pip install surprise

### JJ Baseline (adding RMSE)

# Load the data
train_data = pd.read_csv('archive/interactions_train.csv')
test_data = pd.read_csv('archive/interactions_test.csv')

# Baseline: Average Ratings
allRatings = []
userRatings = defaultdict(list)
recipeRatings = defaultdict(list)

for i in range(len(train_data)):
    allRatings.append(train_data['rating'][i])
    userRatings[train_data['user_id'][i]].append(train_data['rating'][i])
    recipeRatings[train_data['recipe_id'][i]].append(train_data['rating'][i])

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {u: sum(ratings) / len(ratings) for u, ratings in userRatings.items()}
recipeAverage = {r: sum(ratings) / len(ratings) for r, ratings in recipeRatings.items()}

# Apply baseline to test data
predictions = []
for i in range(len(test_data)):
    user = test_data['user_id'][i]
    recipe = test_data['recipe_id'][i]
    if user in userAverage and recipe in recipeAverage:
        predictions.append((userAverage[user] + recipeAverage[recipe]) / 2)
    elif user in userAverage:
        predictions.append(userAverage[user])
    elif recipe in recipeAverage:
        predictions.append(recipeAverage[recipe])
    else:
        predictions.append(globalAverage)

# Add predictions to the test data
test_data['prediction'] = predictions

# Calculate RMSE for Baseline
mse_baseline = mean_squared_error(test_data['rating'], test_data['prediction'])
rmse_baseline = np.sqrt(mse_baseline)

# Define a function to calculate metrics
def calculate_metrics(test_labels, predictions):
    metrics = {}
    metrics['MSE'] = mean_squared_error(test_labels, predictions)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(test_labels, predictions)

    # Binarize predictions and actual ratings for classification metrics
    binary_actuals = (test_labels >= 4)
    binary_preds = (np.array(predictions) >= 4)
    tn, fp, fn, tp = confusion_matrix(binary_actuals, binary_preds).ravel()

    # Classification metrics
    metrics['accuracy'] = accuracy_score(binary_actuals, binary_preds)
    metrics['precision'] = precision_score(binary_actuals, binary_preds)
    metrics['recall'] = recall_score(binary_actuals, binary_preds)
    metrics['F1'] = f1_score(binary_actuals, binary_preds)
    metrics['BER'] = 0.5 * (fp / (fp + tn) + fn / (tp + fn))
    metrics['TP'] = tp
    metrics['TN'] = tn
    metrics['FP'] = fp
    metrics['FN'] = fn

    return metrics

# Calculate baseline metrics
baseline_metrics = calculate_metrics(test_data['rating'], predictions)

# Print baseline metrics
print(baseline_metrics)

### JJ's CODE

### SVD ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score

# Ensure no missing values in critical columns
train_data.dropna(subset=['user_id', 'recipe_id', 'rating'], inplace=True)
test_data.dropna(subset=['user_id', 'recipe_id', 'rating'], inplace=True)

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_data[['user_id', 'recipe_id', 'rating']], reader)

# SVD model and cross-validation
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the algorithm on the full trainset and predict on the testset
trainset = data.build_full_trainset()
svd.fit(trainset)
testset = list(map(lambda x: (x[0], x[1], x[2]), test_data[['user_id', 'recipe_id', 'rating']].values))
predictions = svd.test(testset)

# Calculate RMSE, MSE, and MAE from the predictions
actuals = np.array([pred.r_ui for pred in predictions])
preds = np.array([pred.est for pred in predictions])
mse_svd = mean_squared_error(actuals, preds)
rmse_svd = np.sqrt(mse_svd)
mae_svd = np.mean(np.abs(actuals - preds))

# Classification metrics
# Binarize ratings for classification metrics (ratings of 4 or 5 are considered positive)
binary_actuals = (actuals >= 4)
binary_preds = (preds >= 4)
tn, fp, fn, tp = confusion_matrix(binary_actuals, binary_preds).ravel()
precision = precision_score(binary_actuals, binary_preds)
recall = recall_score(binary_actuals, binary_preds)
f1 = f1_score(binary_actuals, binary_preds)
ber = 0.5 * (fp / (fp + tn) + fn / (tp + fn))
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Results
results = {
    'TP': tp,
    'TN': tn,
    'FP': fp,
    'FN': fn,
    'BER': ber,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'F1': f1,
    'MSE': mse_svd,
    'MAE': mae_svd,
    'RMSE': rmse_svd
}

# Output results
print(results)

# Plotting the distribution of the first latent factor for users
plt.hist(svd.pu[:, 0], bins=30, alpha=0.7)
plt.title("Distribution of the First Latent Factor for Users")
plt.xlabel("Factor Value")
plt.ylabel("Frequency")
plt.show()

"""#### **Linear Regression**"""

### Linear Regression ###

# Calculate the average ratings per user and per recipe
user_avg_rating = train_data.groupby('user_id')['rating'].mean().rename('user_avg_rating')
recipe_avg_rating = train_data.groupby('recipe_id')['rating'].mean().rename('recipe_avg_rating')

# Merge the average ratings into the original datasets
train_data = train_data.join(user_avg_rating, on='user_id').join(recipe_avg_rating, on='recipe_id')
test_data = test_data.join(user_avg_rating, on='user_id').join(recipe_avg_rating, on='recipe_id')

# Check if the new columns exist in the train and test data
print(train_data.columns)
print(test_data.columns)

# Replace NaN values with global average ratings if any
global_avg_rating = train_data['rating'].mean()
train_data['user_avg_rating'].fillna(global_avg_rating, inplace=True)
train_data['recipe_avg_rating'].fillna(global_avg_rating, inplace=True)
test_data['user_avg_rating'].fillna(global_avg_rating, inplace=True)
test_data['recipe_avg_rating'].fillna(global_avg_rating, inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error

# Assuming train_data and test_data are already loaded and preprocessed

# Prepare the features and target for Linear Regression
X_train = train_data[['user_avg_rating', 'recipe_avg_rating']]
y_train = train_data['rating']
X_test = test_data[['user_avg_rating', 'recipe_avg_rating']]
y_test = test_data['rating']

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test data
predictions_lr = lin_reg.predict(X_test)

# Calculate MSE, MAE, and RMSE for Linear Regression
mse_lr = mean_squared_error(y_test, predictions_lr)
mae_lr = mean_absolute_error(y_test, predictions_lr)
rmse_lr = np.sqrt(mse_lr)

# Binarize the predictions and actual ratings for classification metrics
# Assuming a rating of 4 or above is considered positive
binary_actuals = (y_test >= 4)
binary_preds = (predictions_lr >= 4)

# Calculate classification metrics
tn, fp, fn, tp = confusion_matrix(binary_actuals, binary_preds).ravel()
precision_lr = precision_score(binary_actuals, binary_preds)
recall_lr = recall_score(binary_actuals, binary_preds)
f1_lr = f1_score(binary_actuals, binary_preds)
accuracy_lr = accuracy_score(binary_actuals, binary_preds)
ber_lr = 0.5 * (fp / (fp + tn) + fn / (tp + fn))

# Compile results into a dictionary
lr_results = {
    'TP': tp,
    'TN': tn,
    'FP': fp,
    'FN': fn,
    'BER': ber_lr,
    'accuracy': accuracy_lr,
    'precision': precision_lr,
    'recall': recall_lr,
    'F1': f1_lr,
    'MSE': mse_lr,
    'MAE': mae_lr,
    'RMSE': rmse_lr
}

# Output the results
print(lr_results)

### Random Forest Regressor ###

from sklearn.ensemble import RandomForestRegressor

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
predictions_rf = rf.predict(X_test)

# Calculate MSE, MAE, and RMSE
mse_rf = mean_squared_error(y_test, predictions_rf)
mae_rf = mean_absolute_error(y_test, predictions_rf)
rmse_rf = np.sqrt(mse_rf)

# Binarize predictions and actual ratings for classification metrics
binary_actuals_rf = (y_test >= 4)
binary_preds_rf = (predictions_rf >= 4)
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(binary_actuals_rf, binary_preds_rf).ravel()

# Calculate classification metrics
precision_rf = precision_score(binary_actuals_rf, binary_preds_rf)
recall_rf = recall_score(binary_actuals_rf, binary_preds_rf)
f1_rf = f1_score(binary_actuals_rf, binary_preds_rf)
ber_rf = 0.5 * (fp_rf / (fp_rf + tn_rf) + fn_rf / (tp_rf + fn_rf))
accuracy_rf = (tp_rf + tn_rf) / (tp_rf + tn_rf + fp_rf + fn_rf)

# Compile results
results_rf = {
    'TP': tp_rf,
    'TN': tn_rf,
    'FP': fp_rf,
    'FN': fn_rf,
    'BER': ber_rf,
    'accuracy': accuracy_rf,
    'precision': precision_rf,
    'recall': recall_rf,
    'F1': f1_rf,
    'MSE': mse_rf,
    'MAE': mae_rf,
    'RMSE': rmse_rf
}

print(results_rf)

### Model Comparisons

import pandas as pd

# Replace 'lr_results', 'results', and 'results_rf' with the actual dictionaries from your code
comparison_df = pd.DataFrame([baseline_metrics, lr_results, results, results_rf],
                             index=["Baseline", "Linear Regression", "SVD", "Random Forest"])

# Print the DataFrame
print(comparison_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Plotting Accuracy, Precision, Recall, and F1 Score
plt.figure(figsize=(12, 6))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'F1']
sns.barplot(data=comparison_df[metrics_to_plot])
plt.title('Comparison of Accuracy, Precision, Recall, and F1 Score')
plt.ylabel('Score')
plt.xlabel('Models')
plt.xticks(range(len(metrics_to_plot)), metrics_to_plot)
plt.show()

# Plotting MSE and RMSE
plt.figure(figsize=(12, 6))
error_metrics_to_plot = ['MSE', 'RMSE']
sns.barplot(data=comparison_df[error_metrics_to_plot])
plt.title('Comparison of MSE and RMSE')
plt.ylabel('Error')
plt.xlabel('Models')
plt.xticks(range(len(error_metrics_to_plot)), error_metrics_to_plot)
plt.show()

# Plotting BER and Accuracy
plt.figure(figsize=(12, 6))
ber_acc_to_plot = ['BER', 'accuracy']
sns.barplot(data=comparison_df[ber_acc_to_plot])
plt.title('Comparison of BER and Accuracy')
plt.ylabel('Score')
plt.xlabel('Models')
plt.xticks(range(len(ber_acc_to_plot)), ber_acc_to_plot)
plt.show()

