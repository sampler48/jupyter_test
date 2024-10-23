import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt         #  importing some basic libraries to work with the dataset and for visualization



def load_data(filepath):
    """
    Load the dataset into a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df):
    """
    Check for missing values in the DataFrame.
    """
    missing = df.isnull().sum()
    print("Missing Values:\n", missing)

def generate_summary_statistics(df):
    """
    Generate summary statistics for key variables.
    """
    summary = df.describe()
    print("Summary Statistics:\n", summary)

def visualize_distributions(df):
    """
    Visualize distributions of age, balance, credit_score, and estimated_salary.
    """
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(df['age'], kde=True)
    plt.title('Age Distribution')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['balance'], kde=True)
    plt.title('Balance Distribution')
    
    plt.subplot(2, 2, 3)
    sns.histplot(df['credit_score'], kde=True)
    plt.title('Credit Score Distribution')
    
    plt.subplot(2, 2, 4)
    sns.histplot(df['estimated_salary'], kde=True)
    plt.title('Estimated Salary Distribution')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    df = load_data('data/bank_churn.csv')
    
    # Check for missing values
    check_missing_values(df)
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Visualize distributions
    visualize_distributions(df)

if __name__ == "__main__":
    main()
