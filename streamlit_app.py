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
        '1_x': 'Convex',
        '1_f': 'Flat',
        '1_k': 'Knobbed',
        '1_s': 'Sunken'
    }

    # Calculate value counts for cap shapes
    shape_counts = {shape_names[col]: encoded_data[col].sum() for col in columns}

    # Create combined bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(shape_counts.keys(), shape_counts.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'], alpha=0.7)
    ax.set_title('Distribution of Mushroom Cap Shapes')
    ax.set_xlabel('Cap Shape')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, (shape, count) in enumerate(shape_counts.items()):
        ax.text(i, count, str(count), ha='center', va='bottom')

    # Adjust layout and display plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    #3 Diagram Shape
    # Diagram of mushroom Shape
    st.write('Diagram of mushroom cap surfaces')

    # Define columns for cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
    columns = ['2_f', '2_g', '2_s', '2_y']
    surface_names = {
        '2_f': 'Fibrous',
        '2_g': 'Grooves',
        '2_s': 'Smooth',
        '2_y': 'Scaly'
    }

    # Calculate value counts for cap surfaces
    surface_counts = {surface_names[col]: encoded_data[col].sum() for col in columns}

    # Create combined bar plot for cap surfaces
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(surface_counts.keys(), surface_counts.values(), color=['yellow', 'brown', 'grey', 'pink'], alpha=0.7)
    ax.set_title('Distribution of Mushroom Cap Surfaces')
    ax.set_xlabel('Cap Surface')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, (surface, count) in enumerate(surface_counts.items()):
        ax.text(i, count, str(count), ha='center', va='bottom')

    # Adjust layout and display plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    #4 Diagram cap colors
    # Diagram of mushroom cap colors
    st.write('Diagram of mushroom cap colors')

    # Define columns for cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
    columns = ['3_b', '3_c', '3_e', '3_g', '3_n', '3_p', '3_r', '3_u', '3_w', '3_y']
    color_names = {
        '3_b': 'Buff',
        '3_c': 'Cinnamon',
        '3_e': 'Red',
        '3_g': 'Gray',
        '3_n': 'Brown',
        '3_p': 'Pink',
        '3_r': 'Green',
        '3_u': 'Purple',
        '3_w': 'White',
        '3_y': 'Yellow'
    }

    # Calculate value counts for cap colors
    color_counts = {color_names[col]: encoded_data[col].sum() for col in columns}

    # Create combined bar plot for cap colors
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(color_counts.keys(), color_counts.values(), color=['brown', 'burlywood', 'chocolate', 'grey', 'darkgreen', 'pink', 'purple', 'red', 'white', 'yellow'], alpha=0.7)
    ax.set_title('Distribution of Mushroom Cap Colors')
    ax.set_xlabel('Cap Color')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, (color, count) in enumerate(color_counts.items()):
        ax.text(i, count, str(count), ha='center', va='bottom')

    # Adjust layout and display plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    #5 Diagram bruises
    # Diagram of bruises
    st.write('Diagram of mushroom bruises')

    # Define columns for bruises: bruises=t, no_bruises=f
    columns = ['4_t', '4_f']
    bruise_names = {
        '4_t': 'Bruises',
        '4_f': 'No Bruises'
    }

    # Calculate value counts for bruises
    bruise_counts = {bruise_names[col]: encoded_data[col].sum() for col in columns}

    # Create combined bar plot for bruises
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bruise_counts.keys(), bruise_counts.values(), color=['darkblue', 'lightblue'], alpha=0.7)
    ax.set_title('Distribution of Mushroom Bruises')
    ax.set_xlabel('Bruises')
    ax.set_ylabel('Frequency')

    # Add frequency labels on top of bars
    for i, (bruise, count) in enumerate(bruise_counts.items()):
        ax.text(i, count, str(count), ha='center', va='bottom')

    # Adjust layout and display plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    
    # Diagram of mushroom odors
    # Define legend text
legend_text = "Odor Types:\n" \
              "almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s"

    # List of column names
    columns = ['5_a', '5_l', '5_c', '5_y', '5_f', '5_m', '5_n', '5_p', '5_s']
    
    # Calculate value counts for odor types
    odor_counts = {col: encoded_data[col].sum() for col in columns}
    
    # Define colors for each odor type
    odor_colors = {
        '5_a': 'red',   # almond
        '5_l': 'green',  # anise
        '5_c': 'blue',   # creosote
        '5_y': 'orange',  # fishy
        '5_f': 'purple',  # foul
        '5_m': 'yellow',  # musty
        '5_n': 'pink',   # none
        '5_p': 'brown',   # pungent
        '5_s': 'gray'    # spicy
    }
    
    # Create a bar plot for odor types
    fig, ax = plt.subplots(figsize=(15, 5))
    bars = ax.bar(odor_counts.keys(), odor_counts.values(), color=[odor_colors[col] for col in columns], alpha=0.7)
    ax.set_title('Distribution of Mushroom Odor')
    ax.set_xlabel('Odor Type')
    ax.set_ylabel('Frequency')
    
    # Add frequency labels on top of bars
    for odor, count in odor_counts.items():
        ax.text(odor, count, str(count), ha='center', va='bottom')
    
    # Display legend text
    ax.legend(bars, [legend_text], loc='upper left')
    
    # Display plot in Streamlit
    st.write('Diagram of Mushroom Odor')
    st.pyplot(fig)


    # Diagram gill-attachment

    
