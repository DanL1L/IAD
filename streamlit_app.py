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
    st.write('Diagram of mushroom odors')
    
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
    st.pyplot(fig)
    # Define the mappings for each feature and their colors
    features = {
        '6. gill-attachment': {'6_a': 'Attached', '6_d': 'Descending', '6_f': 'Free', '6_n': 'Notched'},
        '7. gill-spacing': {'7_c': 'Close', '7_w': 'Crowded', '7_d': 'Distant'},
        '8. gill-size': {'8_b': 'Broad', '8_n': 'Narrow'},
        '9. gill-color': {'9_k': 'Black', '9_n': 'Brown', '9_b': 'Buff', '9_h': 'Chocolate', '9_g': 'Gray',
                          '9_r': 'Green', '9_o': 'Orange', '9_p': 'Pink', '9_u': 'Purple', '9_e': 'Red',
                          '9_w': 'White', '9_y': 'Yellow'},
        '10. stalk-shape': {'10_e': 'Enlarging', '10_t': 'Tapering'},
        '11. stalk-root': {'11_b': 'Bulbous', '11_c': 'Club', '11_u': 'Cup', '11_e': 'Equal', '11_z': 'Rhizomorphs',
                           '11_r': 'Rooted', '11_?': 'Missing'},
        '12. stalk-surface-above-ring': {'12_f': 'Fibrous', '12_y': 'Scaly', '12_k': 'Silky', '12_s': 'Smooth'},
        '13. stalk-surface-below-ring': {'13_f': 'Fibrous', '13_y': 'Scaly', '13_k': 'Silky', '13_s': 'Smooth'},
        '14. stalk-color-above-ring': {'14_n': 'Brown', '14_b': 'Buff', '14_c': 'Cinnamon', '14_g': 'Gray', '14_o': 'Orange',
                                       '14_p': 'Pink', '14_e': 'Red', '14_w': 'White', '14_y': 'Yellow'},
        '15. stalk-color-below-ring': {'15_n': 'Brown', '15_b': 'Buff', '15_c': 'Cinnamon', '15_g': 'Gray', '15_o': 'Orange',
                                       '15_p': 'Pink', '15_e': 'Red', '15_w': 'White', '15_y': 'Yellow'},
        '16. veil-type': {'16_p': 'Partial', '16_u': 'Universal'},
        '17. veil-color': {'17_n': 'Brown', '17_o': 'Orange', '17_w': 'White', '17_y': 'Yellow'},
        '18. ring-number': {'18_n': 'None', '18_o': 'One', '18_t': 'Two'},
        '19. ring-type': {'19_c': 'Cobwebby', '19_e': 'Evanescent', '19_f': 'Flaring', '19_l': 'Large',
                          '19_n': 'None', '19_p': 'Pendant', '19_s': 'Sheathing', '19_z': 'Zone'},
        '20. spore-print-color': {'20_k': 'Black', '20_n': 'Brown', '20_b': 'Buff', '20_h': 'Chocolate', '20_r': 'Green',
                                  '20_o': 'Orange', '20_u': 'Purple', '20_w': 'White', '20_y': 'Yellow'},
        '21. population': {'21_a': 'Abundant', '21_c': 'Clustered', '21_n': 'Numerous', '21_s': 'Scattered',
                           '21_v': 'Several', '21_y': 'Solitary'},
        '22. habitat': {'22_g': 'Grasses', '22_l': 'Leaves', '22_m': 'Meadows', '22_p': 'Paths', '22_u': 'Urban',
                        '22_w': 'Waste', '22_d': 'Woods'}
    }
    
    # Define colors for each feature
    feature_colors = {
        '6. gill-attachment': {'6_a': 'blue', '6_d': 'orange', '6_f': 'green', '6_n': 'red'},
        '7. gill-spacing': {'7_c': 'blue', '7_w': 'orange', '7_d': 'green'},
        '8. gill-size': {'8_b': 'blue', '8_n': 'orange'},
        '9. gill-color': {'9_k': 'black', '9_n': 'brown', '9_b': 'lightgray', '9_h': 'chocolate', '9_g': 'gray',
                          '9_r': 'green', '9_o': 'orange', '9_p': 'pink', '9_u': 'purple', '9_e': 'red',
                          '9_w': 'white', '9_y': 'yellow'},
        '10. stalk-shape': {'10_e': 'blue', '10_t': 'orange'},
        '11. stalk-root': {'11_b': 'blue', '11_c': 'orange', '11_u': 'green', '11_e': 'red', '11_z': 'purple',
                           '11_r': 'brown', '11_?': 'gray'},
        '12. stalk-surface-above-ring': {'12_f': 'blue', '12_y': 'orange', '12_k': 'green', '12_s': 'red'},
        '13. stalk-surface-below-ring': {'13_f': 'blue', '13_y': 'orange', '13_k': 'green', '13_s': 'red'},
        '14. stalk-color-above-ring': {'14_n': 'brown', '14_b': 'lightgray', '14_c': 'orange', '14_g': 'gray', '14_o': 'green',
                                       '14_p': 'pink', '14_e': 'red', '14_w': 'white', '14_y': 'yellow'},
        '15. stalk-color-below-ring': {'15_n': 'brown', '15_b': 'lightgray', '15_c': 'orange', '15_g': 'gray', '15_o': 'green',
                                       '15_p': 'pink', '15_e': 'red', '15_w': 'white', '15_y': 'yellow'},
        '16. veil-type': {'16_p': 'blue', '16_u': 'orange'},
        '17. veil-color': {'17_n': 'brown', '17_o': 'orange', '17_w': 'white', '17_y': 'yellow'},
        '18. ring-number': {'18_n': 'blue', '18_o': 'orange', '18_t': 'green'},
        '19. ring-type': {'19_c': 'blue', '19_e': 'orange', '19_f': 'green', '19_l': 'red',
                          '19_n': 'purple', '19_p': 'brown', '19_s': 'pink', '19_z': 'gray'},
        '20. spore-print-color': {'20_k': 'black', '20_n': 'brown', '20_b': 'lightgray', '20_h': 'chocolate', '20_r': 'green',
                                  '20_o': 'orange', '20_u': 'purple', '20_w': 'white', '20_y': 'yellow'},
        '21. population': {'21_a': 'blue', '21_c': 'orange', '21_n': 'green', '21_s': 'red',
                           '21_v': 'purple', '21_y': 'pink'},
        '22. habitat': {'22_g': 'blue', '22_l': 'orange', '22_m': 'green', '22_p': 'red',
                        '22_u': 'purple', '22_w': 'brown', '22_d': 'gray'}
    }
    
    # Function to create bar plots for each feature
    def create_bar_plot(feature_name, feature_dict, color_dict):
        # Filter out columns that do not exist in the DataFrame
        existing_columns = {col: feature_dict[col] for col in feature_dict.keys() if col in encoded_data.columns}
        if not existing_columns:
            return None
        
        feature_counts = {existing_columns[col]: encoded_data[col].sum() for col in existing_columns.keys()}
        
        fig, ax = plt.subplots(figsize=(15, 5))
        bars = ax.bar(feature_counts.keys(), feature_counts.values(), color=[color_dict[col] for col in existing_columns.keys()], alpha=0.7)
        ax.set_title(f'Distribution of Mushroom {feature_name}')
        ax.set_xlabel(feature_name.split('.')[1].strip())
        ax.set_ylabel('Frequency')
        
        for i, (feature, count) in enumerate(feature_counts.items()):
            ax.text(i, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    # Iterate through the features and create plots
    for feature_name, feature_dict in features.items():
        st.write(f'Diagram of Mushroom {feature_name.split(".")[1].strip()}')
        fig = create_bar_plot(feature_name, feature_dict, feature_colors[feature_name])
        if fig:
            st.pyplot(fig)
        else:
            st.write(f"No data available for {feature_name}")
            
