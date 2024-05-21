import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load  dataset
data = pd.read_csv('agaricus-lepiota.data', header=None)

# Save Data to a CSV file
encoded_df.to_csv('encoded_data.csv', index=False)

encoded_df.head()

# one-hot encoding
encoded_data = pd.get_dummies(data)

# Save to CSV
encoded_data.to_csv('encoded_data.csv', index=False)

# Data
print('Shape of encoded DataFrame:', encoded_data.shape)
# Convert boolean to int
encoded_data = encoded_data * 1

# Save  CSV file
encoded_data.to_csv('updated_encoded_data.csv', index=False)

# Print nr data
print('Shape of updated encoded DataFrame:', encoded_data.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Load  data
encoded_data = pd.read_csv('updated_encoded_data.csv')

# Check the column names
print(encoded_data.columns)

target_column_name = "0_e"

# Split the data into features and target
X = encoded_data.drop(columns=[target_column_name])
y = encoded_data[target_column_name]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naïve Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Train and evaluate the models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print(f'{name} accuracy: {score:.2f}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
encoded_data = pd.read_csv('updated_encoded_data.csv')

target_column_name = '0_e'

# Split the data into features and target
X = encoded_data.drop(columns=[target_column_name])
y = encoded_data[target_column_name]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Decision Tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train_scaled, y_train)

# Predict the test set
y_pred = model.predict(X_test_scaled)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the Decision Tree model:', accuracy)


from sklearn.model_selection import cross_val_score

# Assuming you've trained a Decision Tree model earlier
decision_tree_model = DecisionTreeClassifier()

# Perform cross-validation to check for overfitting
scores = cross_val_score(decision_tree_model, X, y, cv=5)
print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())
print('Standard deviation of cross-validation scores:', scores.std())

import pandas as pd
# Ciuperci comestibile
edible_count = encoded_data['0_e'].sum()
print('Nr. edible mushrooms:', edible_count)

import pandas as pd
import matplotlib.pyplot as plt

# List of column names
columns = ['0_e', '0_p']

# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Plot each column
for i, column in enumerate(columns):
    value_counts = encoded_data[column].value_counts()
    value_counts.plot(kind='bar', ax=axs[i], color='skyblue', alpha=0.7)
    axs[i].set_title(f'Distribution of {column}')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for index, value in enumerate(value_counts):
        axs[i].text(index, value, str(value), ha='center', va='bottom')

# Adjust layout
plt.tight_layout()
plt.show()
