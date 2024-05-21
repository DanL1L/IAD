import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load dataset
st.title("Clasificarea ciupercilor")
st.write("Proiect Individual (Daniel Lupacescu / Marcela Stratan ) .")

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="data")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    
    # One-hot encoding
    st.write("## Original Data")
    st.write(data.head())

    encoded_data = pd.get_dummies(data)
    encoded_data.to_csv('encoded_data.csv', index=False)
    
    # Save to CSV
    encoded_data = encoded_data * 1
    encoded_data.to_csv('updated_encoded_data.csv', index=False)

    # Display data
    st.write("## Encoded Data")
    st.write(encoded_data.head())

    # Data dimensions
    st.write('Shape of encoded DataFrame:', encoded_data.shape)

    # Splitting the data
    target_column_name = "0_e"
    X = encoded_data.drop(columns=[target_column_name])
    y = encoded_data[target_column_name]
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
    st.write("## Model Performance")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        st.write(f'{name} accuracy: {score:.2f}')

    # Decision Tree Confusion Matrix
    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    st.write('## Confusion Matrix for Decision Tree')
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix for Decision Tree')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.write('Accuracy of the Decision Tree model:', accuracy)

    # Cross-validation scores
    st.write("## Cross-validation scores")
    decision_tree_model = DecisionTreeClassifier()
    scores = cross_val_score(decision_tree_model, X, y, cv=5)
    st.write('Cross-validation scores:', scores)
    st.write('Mean cross-validation score:', scores.mean())
    st.write('Standard deviation of cross-validation scores:', scores.std())

    # Edible mushrooms count
    edible_count = encoded_data['0_e'].sum()
    st.write('Number of edible mushrooms:', edible_count)

    #1 Diagram of edible and poisonous mushrooms
    st.write('Diagram of edible and poisonous mushrooms:')

    # List of column names
    columns = ['0_e', '0_p']

    # Rename columns
    column_names = {'0_e': 'Edible Mushrooms', '0_p': 'Poison Mushrooms'}

    # Calculate value counts
    value_counts = [encoded_data['0_e'].sum(), encoded_data['0_p'].sum()]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(column_names.values(), value_counts, color='skyblue', alpha=0.7)
    ax.set_title('Distribution of Edible and Poisonous Mushrooms')
    ax.set_xlabel('Mushroom Type')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, value in enumerate(value_counts):
        ax.text(i, value, str(value), ha='center', va='bottom')

    # Display plot in Streamlit
    st.pyplot(fig)
    
    #2 Diagram Shape
    
    # Define columns for cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    columns = ['1_b', '1_c', '1_f', '1_k', '1_s', '1_x']

    # Create subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axs = axs.flatten()

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

    # Adjust layout and display plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
