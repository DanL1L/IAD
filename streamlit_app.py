import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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
    # List of column names
    columns = ['0_e', '0_p']

    # Rename columns
    column_names = {'0_e': 'Edible Mushrooms', '0_p': 'Poisonous Mushrooms'}

    # Calculate value counts
    value_counts = [encoded_data['0_e'].sum(), encoded_data['0_p'].sum()]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(column_names.values(), value_counts, color=['green', 'red'], alpha=0.7)
    ax.set_title('Distribution of Edible and Poisonous Mushrooms')
    ax.set_xlabel('Mushroom Type')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, value in enumerate(value_counts):
        ax.text(i, value, str(value), ha='center', va='bottom')

    # Display plot in Streamlit
    st.pyplot(fig)
    
    #2 Diagram Shape
    st.write('Diagram of mushroom cap shapes')

    # Define columns for cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    columns = ['1_b', '1_c', '1_f', '1_k', '1_s', '1_x']
    shape_names = {
        '1_b': 'Bell',
        '1_c': 'Conical',
        '1_f': 'Flat',
        '1_k': 'Knobbed',
        '1_s': 'Sunken',
        '1_x': 'Convex'
    }

    # Calculate value counts for each shape
    shape_counts = {shape_names[col]: encoded_data[col].sum() for col in columns}

    # Create bar plot for mushroom cap shapes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(shape_counts.keys(), shape_counts.values(), color='blue', alpha=0.7)
    ax.set_title('Distribution of Mushroom Cap Shapes')
    ax.set_xlabel('Cap Shape')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, value in enumerate(shape_counts.values()):
        ax.text(i, value, str(value), ha='center', va='bottom')

    # Display plot in Streamlit
    st.pyplot(fig)

    # Function to plot learning curves
    def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    # Plot learning curves for each classifier
    st.write("## Learning Curves")
    
    st.write("### Neural Network Learning Curve")
    plot_learning_curve(MLPClassifier(max_iter=1000), "Neural Network Learning Curve", X, y, cv=5)
    st.pyplot(plt)

    st.write("### Decision Tree Learning Curve")
    plot_learning_curve(DecisionTreeClassifier(), "Decision Tree Learning Curve", X, y, cv=5)
    st.pyplot(plt)

    st.write("### Naive Bayes Learning Curve")
    plot_learning_curve(GaussianNB(), "Naive Bayes Learning Curve", X, y, cv=5)
    st.pyplot(plt)

