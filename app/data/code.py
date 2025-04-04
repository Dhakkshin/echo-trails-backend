codes = [
  {
    "QNo": 1,
    "Code": """
import heapq

def a_star(graph, start, goal, heuristics):
    \"\"\"
    Finds the shortest path from start to goal in a directed weighted graph using A*.

    Args:
        graph: A dictionary representing the graph where keys are nodes and 
               values are lists of (neighbor, edge_cost) tuples.
        start: The starting node.
        goal: The destination node.
        heuristics: A dictionary of heuristic values for each node.

    Returns:
        A list of nodes representing the optimal path from start to goal, 
        or None if no path exists.
    \"\"\"

    open_set = [(0 + heuristics[start], 0, [start])]  # (f_score, g_score, path)
    visited = set()

    while open_set:
        f_score, g_score, path = heapq.heappop(open_set)
        node = path[-1]

        if node == goal:
            return path

        if node in visited:
            continue
        visited.add(node)

        for neighbor, cost in graph.get(node, []):
            new_g_score = g_score + cost
            new_path = list(path)
            new_path.append(neighbor)
            priority = new_g_score + heuristics[neighbor]
            heapq.heappush(open_set, (priority, new_g_score, new_path))

    return None  # No path found

# Example Usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 3), ('E', 2)],
    'C': [('F', 5)],
    'D': [('G', 2)],
    'E': [('G', 1)],
    'F': [('G', 3)]
}
heuristics = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 1,
    'E': 1,
    'F': 0,
    'G': 0
}

start_node = 'A'
goal_node = 'G'

path = a_star(graph, start_node, goal_node, heuristics)

if path:
    print("Optimal path:", path)
else:
    print("No path exists.")
"""
  },
  {
    "QNo": 2,
    "Code": """
import heapq

def a_star_cost(graph, start, target, heuristics):
    \"\"\"
    Determines the minimum path cost from a starting node to a target node 
    in a directed weighted graph using the A* search algorithm.

    Args:
        graph: A dictionary representing the graph where keys are nodes and 
               values are lists of (neighbor, edge_cost) tuples.
        start: The starting node.
        target: The target node.
        heuristics: A dictionary of heuristic values for each node.

    Returns:
        The total path cost from start to target, or \"Path does not exist!\" 
        if no path is found.
    \"\"\"
    open_set = [(0 + heuristics[start], start)]  # (f_score, node)
    cost_so_far = {start: 0}
    visited = set()

    while open_set:
        f_score, current_node = heapq.heappop(open_set)

        if current_node == target:
            return cost_so_far[current_node]

        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, cost in graph.get(current_node, []):
            new_cost = cost_so_far[current_node] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristics[neighbor]
                heapq.heappush(open_set, (priority, neighbor))

    return \"Path does not exist!\"

# Example Usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 3), ('E', 2)],
    'C': [('F', 5)],
    'D': [('G', 2)],
    'E': [('G', 1)],
    'F': [('G', 3)]
}
heuristics = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 1,
    'E': 1,
    'F': 0,
    'G': 0
}

start_node = 'A'
target_node = 'G'

result = a_star_cost(graph, start_node, target_node, heuristics)
print(result)
"""
  },
  {
    "QNo": 3,
    "Code": """
import math

def minimax_alpha_beta(node, depth, maximizing_player, alpha, beta,
                        get_children, get_node_value):
    \"\"\"
    Implements the minimax algorithm with alpha-beta pruning to determine the 
    most favorable investment strategy.

    Args:
        node: The current node in the decision tree.
        depth: The depth of the search tree.
        maximizing_player: A boolean indicating whether the current player is the maximizing player.
        alpha: The alpha value for alpha-beta pruning.
        beta: The beta value for beta pruning.
        get_children: A function that takes a node and returns its children.
        get_node_value: A function that takes a node and returns its value.

    Returns:
        The optimal value for the current node.
    \"\"\"

    if depth == 0:
        return get_node_value(node)

    if maximizing_player:
        max_eval = -math.inf
        for child in get_children(node):
            eval = minimax_alpha_beta(child, depth - 1, False, alpha, beta,
                                       get_children, get_node_value)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = math.inf
        for child in get_children(node):
            eval = minimax_alpha_beta(child, depth - 1, True, alpha, beta,
                                       get_children, get_node_value)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

# --- Example Usage ---
# Define a simple tree structure and evaluation functions
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

def get_children(node):
    return node.children

def get_node_value(node):
    return node.value

# Create the decision tree
#       A
#      / \\
#     B   C
#    /\\   /\\
#   D  E F  G
tree = Node('A', [
    Node('B', [Node('D', [Node(2), Node(5)]), Node('E', [Node(7), Node(1)])]),
    Node('C', [Node('F', [Node(3), Node(9)]), Node('G', [Node(0), Node(6)])])
])

# Run minimax with alpha-beta pruning
depth = 3  # Depth of the tree to search
result = minimax_alpha_beta(tree, depth, True, -math.inf, math.inf,
                            get_children, get_node_value)
print("Optimal value:", result)
"""
  },
  {
    "QNo": 4,
    "Code": """
import math

def minimax_alpha_beta_water_distribution(node, depth, maximizing_player, alpha, beta,
                                            get_children, get_node_value):
    \"\"\"
    Implements the minimax algorithm with alpha-beta pruning to optimize water 
    distribution strategies, including a safety buffer.

    Args:
        node: The current node in the decision tree.
        depth: The depth of the search tree.
        maximizing_player: A boolean indicating whether the current player is the maximizing player.
        alpha: The alpha value for alpha-beta pruning.
        beta: The beta value for beta pruning.
        get_children: A function that takes a node and returns its children.
        get_node_value: A function that takes a node and returns the base water distribution value.

    Returns:
        The optimal value for the current node.
    \"\"\"
    if depth == 0:
        return get_node_value(node) + 10  # Apply safety buffer at leaf nodes

    if maximizing_player:
        max_eval = -math.inf
        for child in get_children(node):
            eval = minimax_alpha_beta_water_distribution(child, depth - 1, False, alpha, beta,
                                                       get_children, get_node_value)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = math.inf
        for child in get_children(node):
            eval = minimax_alpha_beta_water_distribution(child, depth - 1, True, alpha, beta,
                                                       get_children, get_node_value)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

# --- Example Usage ---
# Define a simple tree structure and evaluation functions
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

def get_children(node):
    return node.children

def get_node_value(node):
    return node.value

# Create the decision tree
#       A
#      / \\
#     B   C
#    /\\   /\\
#   D  E F  G
tree = Node('A', [
    Node('B', [Node('D', [Node(2), Node(5)]), Node('E', [Node(7), Node(1)])]),
    Node('C', [Node('F', [Node(3), Node(9)]), Node('G', [Node(0), Node(6)])])
])

# Run minimax with alpha-beta pruning for water distribution
depth = 3  # Depth of the tree to search
result = minimax_alpha_beta_water_distribution(tree, depth, True, -math.inf, math.inf,
                                               get_children, get_node_value)
print("Optimal water distribution value:", result)
"""
  },
  {
    "QNo": 5,
    "Code": """
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handle_missing_data_and_scale(filename):
    \"\"\"
    Loads a dataset, handles missing values, and performs standardization and normalization.

    Args:
        filename: The path to the CSV file.
    \"\"\"

    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Display initial dataset info
    print("--- Initial Dataset Info ---")
    print(df.info())

    # Drop rows with missing values
    df_dropped = df.dropna()
    print("\\n--- Dataset After Dropping Rows with Missing Values ---")
    print(df_dropped)

    # Fill missing values
    df['Employment'].fillna(df['Employment'].mode()[0], inplace=True)
    df['Population'].fillna(df['Population'].mean(), inplace=True)
    df['Income'].fillna(df['Income'].median(), inplace=True)

    print("\\n--- Dataset After Filling Missing Values ---")
    print(df)

    # Display filled values
    print("\\n--- Filled Values ---")
    print("Employment (Mode):", df['Employment'].mode()[0])
    print("Population (Mean):", df['Population'].mean())
    print("Income (Median):", df['Income'].median())

    # Standardization
    scaler = StandardScaler()
    df['Income_standardized'] = scaler.fit_transform(df[['Income']])

    # Normalization
    scaler = MinMaxScaler()
    df['Income_normalized'] = scaler.fit_transform(df[['Income']])

    print("\\n--- Dataset After Standardization and Normalization ---")
    print(df)

    # Display standardized and normalized values
    print("\\n--- Standardized Income Values ---")
    print(df[['Income_standardized']].head())  # Display first few rows

    print("\\n--- Normalized Income Values ---")
    print(df[['Income_normalized']].head())  # Display first few rows

# --- Example Usage ---
filename = 'loan_dataset.csv'  # Replace with your CSV file name
handle_missing_data_and_scale(filename)
"""
  },
  {
    "QNo": 6,
    "Code": """
import pandas as pd
from scipy.stats import linregress

def perform_linear_regression(filename, x_value_to_predict):
    \"\"\"
    Performs linear regression analysis on a dataset and predicts a value.

    Args:
        filename: The path to the CSV file.
        x_value_to_predict: The x value for which to predict the y value.
    \"\"\"

    # Read the data from the CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Check if the DataFrame has the required columns
    if 'x' not in df.columns or 'y' not in df.columns:
        print("Error: The CSV file must contain columns named 'x' and 'y'.")
        return

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df['x'], df['y'])

    # Calculate the estimated value at x
    estimated_y = slope * x_value_to_predict + intercept

    # Print the results
    print("Slope:", round(slope, 4))
    print("Intercept:", round(intercept, 4))
    print("Estimated value at x =", x_value_to_predict, ":", round(estimated_y, 4))

# --- Example Usage ---
filename = 'regression_data.csv'  # Replace with your CSV file name
x_value_to_predict = 10
perform_linear_regression(filename, x_value_to_predict)
"""
  },
  {
    "QNo": 7,
    "Code": """
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

def calculate_rmse(filename):
    \"\"\"
    Calculates the Root Mean Squared Error (RMSE) of a simple linear regression.

    Args:
        filename: The path to the CSV file.
    \"\"\"

    # Read the data from the CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Check if the DataFrame has the required columns
    if 'x' not in df.columns or 'y' not in df.columns:
        print("Error: The CSV file must contain columns named 'x' and 'y'.")
        return

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df['x'], df['y'])

    # Calculate predicted y values
    predicted_y = slope * df['x'] + intercept

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df['y'], predicted_y))

    # Print the RMSE
    print("RMSE:", round(rmse, 3))

# --- Example Usage ---
filename = 'regression_data.csv'  # Replace with your CSV file name
calculate_rmse(filename)
"""
  },
  {
    "QNo": 8,
    "Code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_logistic_regression_churn(filename):
    \"\"\"
    Reads a dataset, fits a logistic regression model to predict customer churn,
    and evaluates the model's performance.

    Args:
        filename: The path to the CSV file.
    \"\"\"

    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Prepare the data
    X = df[['age', 'sex', 'account_length', 'num_products', 'credit_card', 'activity_level', 'estimated_salary']]  # Features
    y = df['Churn']  # Target variable
    X = pd.get_dummies(X, columns=['sex'], drop_first=True)  # Convert 'sex' to numerical

    # Handle missing values (basic imputation for demonstration)
    X = X.fillna(X.mean())  # Fill missing values with the mean

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC-ROC

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    confusion = confusion_matrix(y_test, y_pred)

    # Output the evaluation metrics
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1 Score:", round(f1, 2))
    print("AUC-ROC Score:", round(auc_roc, 2))
    print("Confusion Matrix:\\n", confusion)

# --- Example Usage for Q8 ---
filename_q8 = 'customer_churn.csv'  # Replace with your CSV file name
evaluate_logistic_regression_churn(filename_q8)
"""
  },
  {
    "QNo": 9,
    "Code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def evaluate_logistic_regression_promotion(filename):
    \"\"\"
    Loads a dataset, applies logistic regression to predict employee promotion,
    and calculates the model's performance.

    Args:
        filename: The path to the CSV file.
    \"\"\"
    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Prepare the data
    X = df[['hours_worked']]  # Feature
    y = df['promotion']  # Target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Output the evaluation metrics
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 2))
    print("Accuracy:", round(accuracy, 2))
    print("Confusion Matrix:\\n", confusion)

# --- Example Usage for Q9 ---
filename_q9 = 'employee_performance.csv'  # Replace with your CSV file name
evaluate_logistic_regression_promotion(filename_q9)
"""
  },
  {
    "QNo": 10,
    "Code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_svm_titanic(filename):
    \"\"\"
    Loads a CSV file containing passenger details, preprocesses the data,
    trains an SVM classifier, and evaluates its performance.

    Args:
        filename: The path to the CSV file.
    \"\"\"

    # Load the dataset
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Prepare the data
    # Select features and target
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')  # Impute missing 'Age' with mean
    X['Age'] = imputer.fit_transform(X[['Age']])

    imputer_embarked = SimpleImputer(strategy='most_frequent') # Impute missing 'Embarked' with mode
    X['Embarked'] = imputer_embarked.fit_transform(X[['Embarked']])

    # Encode categorical variables
    le_sex = LabelEncoder()
    X['Sex'] = le_sex.fit_transform(X['Sex'])

    le_embarked = LabelEncoder()
    X['Embarked'] = le_embarked.fit_transform(X['Embarked'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM classifier
    model = SVC(kernel='linear')  # You can experiment with different kernels
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Output the evaluation metrics
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1-score:", round(f1, 2))
    print("Confusion Matrix:\\n", confusion)

# --- Example Usage for Q10 ---
filename_q10 = 'titanic.csv'  # Replace with your CSV file name
evaluate_svm_titanic(filename_q10)
"""
  },
  {
    "QNo": 11,
    "Code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_svm_loan_approval(filename):
    \"\"\"
    Loads a dataset, trains an SVM classifier with a polynomial kernel,
    and computes evaluation metrics. Handles single-class datasets.

    Args:
        filename: The path to the CSV file.
    \"\"\"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Prepare the data
    # Assuming the dataset has features and a target variable
    # You'll need to adjust these lines to match your dataset's structure
    # For example, if your target column is named 'Loan_Approved'
    X = df.drop('Loan_Status', axis=1)  # Features (assuming 'Loan_Status' is the target)
    y = df['Loan_Status']  # Target variable

    # Handle the single class scenario
    if len(y.unique()) <= 1:
        print("Error: Dataset contains only one class. SVM cannot be trained.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM classifier
    model = SVC(kernel='poly')  # Using a polynomial kernel
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Output the evaluation metrics
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1-score:", round(f1, 2))

# --- Example Usage for Q11 ---
filename_q11 = 'loan_approval.csv'  # Replace with your CSV file name
evaluate_svm_loan_approval(filename_q11)
"""
  },
  {
    "QNo": 12,
    "Code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def predict_product_sold_out(filename):
    \"\"\"
    Loads a dataset, preprocesses it, trains a Gaussian Naïve Bayes model to 
    classify whether a product is sold out, and evaluates its performance.

    Args:
        filename: The path to the CSV file.
    \"\"\"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Data Preprocessing ---

    # Handle missing values
    # Select numerical columns for imputation
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer_numerical = SimpleImputer(strategy='mean')  # Or 'median', etc.
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    # Select categorical columns for imputation
    categorical_cols = df.select_dtypes(include=['object']).columns
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is of object type (usually strings)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Scale numerical features (important for some Naïve Bayes variants, though GaussianNB is less sensitive)
    numerical_cols_to_scale = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])


    # --- Model Training and Evaluation ---

    # Prepare data for modeling
    # Assuming 'Sold_Out' is the target variable.  
    # ***You MUST adjust this to the correct name of your target column***
    X = df.drop('Sold_Out', axis=1)  # Features
    y = df['Sold_Out']  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gaussian Naïve Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred) # Get the report as a string

    # Output results
    print("Accuracy:", round(accuracy, 2))
    print("Confusion Matrix:\\n", confusion)
    print("Classification Report:\\n", classification_report_str)

# --- Example Usage for Q12 ---
filename_q12 = 'product_sales.csv'  # Replace with your CSV file name
predict_product_sold_out(filename_q12)"""
   },
   {
        "QNo": 13,
        "Code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def predict_customer_churn(filename):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Data Preprocessing ---

    # Handle missing values
    # Select numerical columns for imputation
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer_numerical = SimpleImputer(strategy='mean')  # Or 'median', etc.
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    # Handle missing values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # Standardize numerical features
    numerical_cols_to_scale = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])

    # --- Model Training and Evaluation ---

    # Prepare data for modeling
    # Assuming 'Churn' is the target variable.  
    # ***You MUST adjust this to the correct name of your target column***
    X = df.drop('Churn', axis=1)  # Features
    y = df['Churn']  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gaussian Naïve Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    # Output results
    print("Accuracy:", round(accuracy, 2))
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification_report_str)

# --- Example Usage for Q13 ---
filename_q13 = 'customer_data.csv'  # Replace with your CSV file name
predict_customer_churn(filename_q13)"""
    },
    {
        "QNo": 14,
        "Code": """import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def categorize_spending_behavior(filename, n_clusters, distance_threshold):
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Data Preprocessing ---

    # Map spending score names to numerical values
    spending_map = {'Low': 0, 'Average': 1, 'High': 2}
    df['Spending Score'] = df['Spending Score'].map(spending_map)

    # Handle missing values (basic imputation for demonstration)
    df = df.fillna(df.mean())  # Fill missing values with the mean

    # Select features for clustering
    features = ['Age', 'Work Experience', 'Spending Score', 'Family Size']
    X = df[features]

    # Scale the features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means Clustering ---

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # You can adjust random_state
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # --- Outlier Detection ---

    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((X_scaled - kmeans.cluster_centers_[df['Cluster']])**2, axis=1))

    # Identify outliers
    df['Category'] = ['Outlier' if dist > distance_threshold else f'Cluster {cluster + 1}' 
                        for cluster, dist in zip(df['Cluster'], distances)]

    # --- Output ---

    for index, row in df.iterrows():
        print(f"Customer {index + 1}: {row['Category']}")

# --- Example Usage for Q14 ---
filename_q14 = 'customer_records.csv'  # Replace with your CSV file name
n_clusters_q14 = 3  # Example: You might need to determine the optimal number of clusters
distance_threshold_q14 = 2.0  # Example: You'll need to determine an appropriate threshold

categorize_spending_behavior(filename_q14, n_clusters_q14, distance_threshold_q14)"""
    },
    {
        "QNo": 15,
        "Code": """import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def classify_weather_patterns(filename):

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Data Preprocessing ---

    # Select features for clustering
    X = df[['TEMPERATURE', 'HUMIDITY']]

    # Handle missing values (basic imputation for demonstration)
    X = X.fillna(X.mean())

    # Scale the features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means Clustering ---

    # Apply K-Means with 4 centroids
    kmeans = KMeans(n_clusters=4, random_state=42)  # You can adjust random_state
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # --- Prediction for New Values ---

    while True:
        try:
            temperature = float(input("Enter temperature: "))
            humidity = float(input("Enter humidity: "))
            break
        except ValueError:
            print("Invalid input. Please enter numerical values.")

    # Scale the new input
    new_data = pd.DataFrame({'TEMPERATURE': [temperature], 'HUMIDITY': [humidity]})
    new_data_scaled = scaler.transform(new_data)  # Use the SAME scaler

    # Predict the cluster
    predicted_cluster = kmeans.predict(new_data_scaled)[0]

    # --- Output ---

    print(f"The weather condition belongs to cluster: {predicted_cluster + 1}")  # 1-based index

# --- Example Usage for Q15 ---
filename_q15 = 'weather_data.csv'  # Replace with your CSV file name
classify_weather_patterns(filename_q15)"""
    },
    {
        "QNo": 16,
        "Code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier  # For a basic neural network

def predict_student_pass_fail(filename):


    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Data Preprocessing ---

    # Handle missing values (basic imputation)
    df = df.fillna(df.mean())  # Or use other strategies if needed

    # Encode target variable ('Pass/Fail' or similar)
    # Assuming the target column is something like 'Result' or 'Pass/Fail'
    # ***You MUST adjust this to the correct name of your target column***
    le = LabelEncoder()
    df['Result'] = le.fit_transform(df['Result'])  # Encodes 'Pass'/'Fail' to 0/1

    # Normalize features
    features = ['Math', 'Science', 'History']  # Adjust if your column names are different
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df['Result']  # Target variable (pass/fail)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Neural Network Training ---

    # Train the neural network
    # MLPClassifier is a basic Multi-layer Perceptron (a type of neural network)
    model = MLPClassifier(hidden_layer_sizes=(10, 5),  # Example: 2 hidden layers
                          activation='relu',  # Common activation function
                          solver='adam',      # Optimization algorithm
                          max_iter=500,       # Maximum iterations
                          random_state=42)
    model.fit(X_train, y_train)

    # --- Prediction for New Scores ---

    while True:
        try:
            math_score = float(input("Enter Math score: "))
            science_score = float(input("Enter Science score: "))
            history_score = float(input("Enter History score: "))
            break
        except ValueError:
            print("Invalid input. Please enter numerical values.")

    # Scale the new input
    new_data = pd.DataFrame({'Math': [math_score], 'Science': [science_score], 'History': [history_score]})
    new_data_scaled = scaler.transform(new_data)  # Use the SAME scaler

    # Predict the result
    prediction = model.predict(new_data_scaled)[0]

    # --- Output ---

    if prediction == 1:  # Assuming 1 represents 'Pass'
        print("Predicted Result: Pass")
    else:
        print("Predicted Result: Fail")

# --- Example Usage for Q16 ---
filename_q16 = 'student_scores.csv'  # Replace with your CSV file name
predict_student_pass_fail(filename_q16)"""
    },
    {
        "QNo": 17,
        "Code": """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np

def gini_impurity(y):

    if len(y) == 0:
        return 0
    probabilities = pd.Series(y).value_counts() / len(y)
    gini = 1 - np.sum(probabilities**2)
    return gini

def predict_credit_risk(filename):

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Data Preprocessing ---

    # Handle missing values
    # Select numerical columns for imputation
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer_numerical = SimpleImputer(strategy='mean')  # Or 'median', etc.
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    # Select categorical columns for imputation
    categorical_cols = df.select_dtypes(include=['object']).columns
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Prepare data for modeling
    # Assuming 'Credit_Risk' is the target variable.  
    # ***You MUST adjust this to the correct name of your target column***
    X = df.drop('Credit_Risk', axis=1)  # Features
    y = df['Credit_Risk']  # Target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Gini Impurity Calculation ---

    # Calculate Gini impurity before training
    gini_before = gini_impurity(y_train)
    print("Gini Impurity Before Training:", round(gini_before, 4))

    # --- Model Training ---

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # --- Gini Impurity Calculation (After Training - on predictions) ---

    # Get predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate Gini impurity of the predictions
    gini_after = gini_impurity(y_pred)
    print("Gini Impurity After Training (on predictions):", round(gini_after, 4))

    # --- Model Evaluation (Basic) ---

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 2))


# --- Example Usage for Q17 ---
filename_q17 = 'credit_risk_data.csv'  # Replace with your CSV file name
predict_credit_risk(filename_q17)"""
    },
    {
        "QNo": 18,
        "Code": """import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def analyze_loan_applications(filename):


    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # --- Identify and Report Missing Data ---

    print("--- Missing Values Count per Column ---")
    print(df.isnull().sum())  # Count missing values in each column

    # --- Handle Missing Data ---

    # Remove rows with missing values
    df_dropped = df.dropna()
    print("\n--- Dataset After Dropping Rows with Missing Values ---")
    print(df_dropped)

    # Fill missing values
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Term'].fillna(df['Loan_Term'].mode()[0], inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Loan_Status'].fillna(df['Loan_Status'].mode()[0], inplace=True)

    # --- Standardization and Normalization ---

    # Standardization (Z-score normalization)
    scaler_standard = StandardScaler()
    df['LoanAmount_standardized'] = scaler_standard.fit_transform(df[['LoanAmount']])

    # Normalization (Min-Max scaling)
    scaler_minmax = MinMaxScaler()
    df['LoanAmount_normalized'] = scaler_minmax.fit_transform(df[['LoanAmount']])

    # --- Display Datasets After Handling Missing Data ---

    print("\n--- Dataset After Handling Missing Data ---")
    print(df)

    # --- Display Standardized and Normalized Values ---

    print("\n--- Standardized LoanAmount Values ---")
    print(df[['LoanAmount_standardized']].head())

    print("\n--- Normalized LoanAmount Values ---")
    print(df[['LoanAmount_normalized']].head())

# --- Example Usage for Q18 ---
filename_q18 = 'loan_applications.csv'  # Replace with your CSV file name
analyze_loan_applications(filename_q18)"""
    }
] 