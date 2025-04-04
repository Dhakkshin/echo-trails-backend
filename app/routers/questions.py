from fastapi import APIRouter, HTTPException
from app.data.code import codes

router = APIRouter()

@router.get("/")
async def get_questions():
    questions = [
  {
    "Q1": "Alex is working on a navigation system that finds the shortest path between two locations using the $A^{*}$ algorithm. [cite: 1, 2] Given a directed graph where nodes represent locations and edges represent paths with weights, along with heuristic values for each node, write a program to determine the optimal path from a starting node to a destination node. [cite: 2, 3, 4] If a path exists, print the sequence of nodes in the order they should be visited; otherwise, print that no path exists. [cite: 3, 4]"
  },
  {
    "Q2": "Implement the $A^{*}$ search algorithm to determine the minimum path cost from a starting node to a target node in a directed weighted graph. [cite: 5, 6] Each node in the graph has a list of neighbors with corresponding edge costs, and each node also has an associated heuristic value estimating its cost to the goal. [cite: 5, 6] If a path exists, output the total path cost; otherwise, output \\\"Path does not exist!\\\". [cite: 6]"
  },
  {
    "Q3": "George, a financial analyst, is working on optimizing the distribution of funds across various investment portfolios. [cite: 7, 8, 9] Each investment scenario is represented as a node in a binary decision tree, where each node decides whether to increase or decrease investment in a particular portfolio. [cite: 7, 8, 9] George aims to use a minimax algorithm with alpha-beta pruning to determine the most favorable investment strategy, maximizing returns at various decision points. [cite: 9]"
  },
  {
    "Q4": "Julia, an environmental scientist, is working on optimizing water distribution strategies during periods of scarcity. [cite: 10, 11, 12, 13] She is using a decision-making tool that evaluates various strategies for allocating water to different regions based on predicted need and availability. [cite: 10, 11, 12, 13] To improve the strategy, Julia implements a minimax algorithm with alpha-beta pruning within a decision tree, where each node represents a decision on water distribution. [cite: 12] Additionally, Julia introduces a flat increase of 10 units of water at each node as a safety buffer, ensuring each area has an emergency reserve. [cite: 13]"
  },
  {
    "Q5": "Vino is a data analyst working with a loan dataset that contains information about urban and rural areas. [cite: 14, 15, 16, 17] The dataset has missing values that need to be addressed. [cite: 15, 16, 17] Vino is also tasked with filling in missing values for the \\\"Employment,\\\" \\\"Population,\\\" and \\\"Income\\\" columns. [cite: 16] Additionally, Vino needs to perform data standardization and normalization for the \\\"Income\\\" column. [cite: 17, 18, 19, 20] Handling Missing Data and Standardization:  Load the CSV dataset, and drop rows with missing values. [cite: 18, 19, 20] Fill categorical \\\"Employment\\\" with mode, \\\"Population\\\" with mean, and \\\"Income\\\" with median.  Display filled values. [cite: 19, 20] Standardize \\\"Income_filled_median\\\" using StandardScaler and normalize using MinMaxScaler. [cite: 20]"
  },
  {
    "Q6": "Lora, a data analyst, is working on a project involving linear regression analysis. [cite: 21, 22, 23, 24, 25, 26, 27] She has a dataset stored in a CSV file, consisting of two columns: 'x' and 'y'. [cite: 22, 23, 24, 25, 26, 27] Lora needs to perform linear regression analysis on this dataset to understand the relationship between $x^{\\prime}$ and 'y'. [cite: 23, 24, 25, 26, 27] To facilitate this analysis, you are tasked with developing a Python program. [cite: 24, 25, 26, 27] The program will prompt the user to input the filename of the dataset. [cite: 25, 26, 27] It will then read the data from the CSV file, execute linear regression analysis using the Scipy library, and calculate the slope, intercept, and estimated value at $x=10$. [cite: 26, 27] Finally, it will print these results rounded to four decimal places. [cite: 27]"
  },
  {
    "Q7": "Paul, a data scientist, is working on a project that involves evaluating the performance of a simple linear regression algorithm. [cite: 28, 29, 30, 31, 32, 33] He has a dataset stored in a CSV file, comprising two columns: 'x' and 'y'. [cite: 29, 30, 31, 32, 33] Paul aims to calculate the root mean squared error (RMSE) of the regression algorithm applied to this dataset. [cite: 30, 31, 32, 33] To assist Paul, you are tasked with creating a Python program. [cite: 31, 32, 33] The program will prompt the user to input the filename of the dataset. [cite: 32, 33] Subsequently, it will read the data from the CSV file, compute the RMSE using a simple linear regression algorithm, and output the RMSE value rounded to three decimal places. [cite: 33]"
  },
  {
    "Q8": "A data scientist is working on a customer churn prediction model for a subscription-based service. [cite: 34, 35, 36, 37, 38] The dataset contains various attributes of customers, such as age, sex, account length, number of products used, credit card status, activity level, and estimated salary. [cite: 35, 36, 37, 38] The target variable Churn indicates whether a customer has canceled their subscription (1) or not (0). [cite: 36, 37, 38] Write a program that reads the dataset from a CSV file, fits a logistic regression model to predict whether a customer will churn based on the given attributes, and evaluates the model's performance using accuracy, precision, recall, F1 score, AUC-ROC score, and confusion matrix. [cite: 37, 38] The program should output the evaluation metrics rounded to two decimal places. [cite: 38]"
  },
  {
    "Q9": "David, an HR manager, is analyzing employee performance using a dataset that contains hours_worked and a binary target promotion (1 if promoted, 0 otherwise). [cite: 39, 40, 41, 42] He needs to predict whether an employee will be promoted based on the number of hours they worked. [cite: 40, 41, 42] Write a program that loads the dataset from a CSV file, applies logistic regression using hours_worked as the feature, and calculates the model's precision, recall, F1 score, accuracy, and confusion matrix. [cite: 41, 42] The program should output the precision score rounded to four decimal places, along with other metrics. [cite: 42]"
  },
  {
    "Q10": "Emily is working on a machine learning project where she needs to classify passenger survival based on Titanic dataset features. [cite: 43, 44, 45] Write a program to load a CSV file containing passenger details, preprocess the data by encoding categorical values and handling missing values, train an SVM classifier, and evaluate its performance using accuracy, precision, recall, F1-score, and a confusion matrix. [cite: 44, 45] The program should read the CSV filename from user input and output the model's evaluation metrics. [cite: 45]"
  },
  {
    "Q11": "Arjun, a data analyst, wants to automate the process of evaluating a loan approval model using an SVM classifier. [cite: 46, 47, 48] Write a program to load a dataset from a given filename, train an SVM classifier with a polynomial kernel, and compute evaluation metrics such as accuracy, precision, recall, and F1-score. [cite: 47, 48] The program should handle cases where the dataset contains only one class and exit gracefully with an error message. [cite: 48]"
  },
  {
    "Q12": "Emma, a retail analyst, needs to predict whether a product will be sold out based on historical sales data. [cite: 49, 50, 51, 52] Write a program that loads a dataset from a user-provided CSV file, preprocesses it by handling missing values, encoding categorical features, and scaling numerical features. [cite: 50, 51, 52] Then, train a Gaussian Naïve Bayes model to classify whether a product is sold out and evaluate its performance using accuracy, confusion matrix, and classification report. [cite: 51, 52]"
  },
  {
    "Q13": "Liam, a customer retention specialist, wants to predict whether a customer will churn based on their demographics and spending behavior. [cite: 52, 53, 54, 55] Write a program that loads a dataset from a user-provided CSV file, preprocesses it by handling missing values and standardizing numerical features. [cite: 53, 54, 55] Then, train a Gaussian Naïve Bayes model to classify whether a customer will churn and evaluate its performance using accuracy, confusion matrix, and classification report. [cite: 54, 55]"
  },
  {
    "Q14": "Emma is analyzing customer demographics to categorize spending behaviors. [cite: 56, 57, 58, 59] Each customer record includes age, work experience, spending score, and family size. [cite: 56, 57, 58, 59] The spending score is mapped to numerical values: Low → 0, Average → 1, High → 2. Given N customer records, Emma wants to classify them into K clusters using the K-Means algorithm. [cite: 57, 58, 59] Any record whose Euclidean distance from its cluster center exceeds a given threshold T is marked as an outlier. [cite: 58, 59] Write a program to determine the cluster number (1-based index) for each customer or mark them as \\\"Outlier\\\". [cite: 59]"
  },
  {
    "Q15": "Aryan is analyzing climate patterns using a dataset that includes WEATHER_ID, TEMPERATURE, HUMIDITY, and CATEGORY. [cite: 60, 61, 62, 63] He wants to apply k-means clustering with four centroids to classify different weather patterns based on temperature and humidity. [cite: 61, 62, 63] Write a program to read the dataset, apply k-means clustering using TEMPERATURE and HUMIDITY, and allow Aryan to input new values to predict the corresponding weather condition cluster. [cite: 62, 63]"
  },
  {
    "Q16": "Athulya aims to create a program that predicts whether a student will pass or fail based on their scores in math, science, and history. [cite: 63, 64, 65, 66, 67] She loads a dataset of scores and pass/fail statuses from a CSV file and trains a neural network using backpropagation. [cite: 64, 65, 66, 67] After normalization and mapping of pass/fail labels, the program trains the network to minimize prediction errors. [cite: 65, 66, 67] Athulya then prompts the user to input scores for math, science, and history. [cite: 66, 67] Using the trained network, the program predicts the student's result as pass or fail. [cite: 67]"
  },
  {
    "Q17": "Prawin is a financial analyst working for a credit risk assessment company. [cite: 68, 69, 70, 71, 72] His task is to develop a classifier that predicts the credit risk of individuals based on their demographic and financial data. [cite: 69, 70, 71, 72] The company has provided him with a dataset containing information about individuals' education, employment status, and credit risk. [cite: 70, 71, 72] Prawin needs to build a decision tree classifier to predict credit risk. [cite: 71, 72] Additionally, he wants to calculate the Gini impurity of the dataset to assess its purity before and after training the model. [cite: 72]"
  },
  {
    "Q18": "As a financial analyst, Ramesh is responsible for evaluating loan applications. [cite: 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83] He has received a dataset containing details of applicants, including demographics, income levels, loan amounts, and credit histories. [cite: 74, 75, 76, 77, 78, 79, 80, 81, 82, 83] His objective is to preprocess the dataset by handling missing values, applying different imputation techniques, and scaling loan amounts for further analysis. [cite: 75, 76, 77, 78, 79, 80, 81, 82, 83] Identify and Report Missing Data:  Read a CSV file containing loan application records. [cite: 76, 77, 78, 79, 80, 81, 82, 83] Identify missing values in each column and report the count of missing values. [cite: 77, 78, 79, 80, 81, 82, 83] Handle Missing Data:  Remove rows that contain missing values. [cite: 78, 79, 80, 81, 82, 83] Fill missing values in \\\"LoanAmount\\\" using the mean. [cite: 78, 79, 80, 81, 82, 83] Fill missing values in \\\"Loan_Term\\\" using the most frequent value. [cite: 79, 80, 81, 82, 83] For categorical columns like \\\"Gender,\\\" \\\"Credit_History,\\\" and \\\"Loan_Status,\\\" replace missing values with the mode. [cite: 80, 81, 82, 83] Standardization and Normalization:  Standardize the \\\"LoanAmount\\\" column using Z-score normalization (StandardScaler). [cite: 81, 82, 83] Normalize the \\\"LoanAmount\\\" column using Min-Max scaling (MinMaxScaler). [cite: 81, 82, 83] Display Datasets After Handling Missing Data:  Show the dataset after dropping rows with missing values. [cite: 82, 83] Display the standardized and normalized values of the \\\"LoanAmount\\\" column. [cite: 83]"
  }
]
    return questions

@router.get("/{qno}")
async def get_answer(qno: int):
    """Get answer code for a specific question number"""
    for answer in codes:
        if answer["QNo"] == qno:
            return answer
    raise HTTPException(status_code=404, detail=f"Answer for question {qno} not found")

