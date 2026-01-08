"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- 
- 
- 
- 

Dataset: [Name of your dataset]
Predicting: [What you're predicting]
Features: [List your features]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  

# TODO: Update this with your actual filename
DATA_FILE = 'banana_quality.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    data = pd.read_csv(filename)
    print("=== Banana Quality Data ===")
    print(f"\nFirst 5 rows: ")
    print(data.head())
    print(f"\nData Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"\nBasic statistics:")
    print(data.describe())
    print(f"\nColumn names: {list(data.columns)}")
    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    fig.suptitle('Banana Features vs Quality', fontsize = 16, fontweight = 'bold')

    axes[0, 0].scatter(data['Size'], data['Quality'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Size(m)')
    axes[0, 0].set_ylabel('Quality(cat)')
    axes[0, 0].set_title('Size vs Quality')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(data['Weight'], data['Quality'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Weight (kg)')
    axes[0, 1].set_ylabel('Quality (cat)')
    axes[0, 1].set_title('Weight vs Quality')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].scatter(data['Sweetness'], data['Quality'], color='red', alpha=0.6)
    axes[0, 2].set_xlabel('Sweetness(index)')
    axes[0, 2].set_ylabel('Quality')
    axes[0, 2].set_title('Sweetness(index) vs Quality')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].scatter(data['Softness'], data['Quality'], color='yellow', alpha=0.6)
    axes[1, 0].set_xlabel('Softness(index)')
    axes[1, 0].set_ylabel('Quality')
    axes[1, 0].set_title('Softness(index) vs Quality')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(data['HarvestTime'], data['Quality'], color='purple', alpha=0.6)
    axes[1, 1].set_xlabel('HarvestTime(time)')
    axes[1, 1].set_ylabel('Quality')
    axes[1, 1].set_title('HarvestTime(time) vs Quality')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].scatter(data['Ripeness'], data['Quality'], color='cyan', alpha=0.6)
    axes[1, 2].set_xlabel('Ripeness(index)')
    axes[1, 2].set_ylabel('Quality')
    axes[1, 2].set_title('Ripeness(index) vs Quality')
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 0].scatter(data['Acidity'], data['Quality'], color='black', alpha=0.6)
    axes[2, 0].set_xlabel('Acidity(index)')
    axes[2, 0].set_ylabel('Quality')
    axes[2, 0].set_title('Acidity(index) vs Quality')
    axes[2, 0].grid(True, alpha=0.3)

    # Hint: Use subplots like in Part 2!
    plt.tight_layout()
    plt.savefig('banana_feature.png', dpi =300, bbox_inches ='tight')
    print("\n✓ Feature plots saved as 'banana_feature.png'")
    plt.show()

def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    feature_columns = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']

    target_columns = ['Quality']

    X = data[feature_columns]

    y = data[target_columns]

    print(f"\n=== Feature Preparation===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature Columns: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)

    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples ")
    print(f"Testing Set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test
    


def train_model(X_train, y_train, feature_names=None):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    if feature_names is None:
        feature_names = list(X_train.columns)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"\n=== Model Training Complete ===")

    # Normalize shapes for printing (coef_ may be 2D when y is a DataFrame)
    coefs = np.array(model.coef_).ravel()
    intercept = model.intercept_
    try:
        intercept = float(np.array(intercept).ravel()[0])
    except Exception:
        intercept = float(intercept)

    print(f"Intercept: {intercept:.2f}")

    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name}: {coef:.2f}")

    print(f"\nEquation:")
    equation = f"Quality = "
    for i, (name, coef) in enumerate(zip(feature_names, coefs)):
        sign = "" if coef >= 0 else "- "
        term = f"{abs(coef):.2f} x {name}"
        if i == 0:
            equation += f"{sign}{term}"
        else:
            equation += f" + ({sign}{term})"
    equation += f" + {intercept:.2f}"
    print(equation)
    return model




def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"\n=== Model Performance ===")
    print(f"R^2 Score: {r2:.4f}")
    print(f" -> Model explains {r2*100:.2f}% of quality variation")

    print(f"\nRoot Mean Squared Error: {rmse:.2f}")
    print(f" -> On average, predictions are off by {rmse:.2f}")

    print(f"\n=== Feature Importance ===")
    if feature_names is None:
        try:
            feature_names = list(X_test.columns)
        except Exception:
            feature_names = [f"x{i}" for i in range(len(np.array(model.coef_).ravel()))]

    coefs = np.array(model.coef_).ravel()
    feature_importance = list(zip(feature_names, np.abs(coefs)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    return predictions

    


def make_prediction(model, size=12.0, weight=0.25, sweetness=5.0, softness=5.0, harvesttime=7.0, ripeness=5.0, acidity=1.0):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    banana_features = pd.DataFrame([[size, weight, sweetness, softness, harvesttime, ripeness, acidity]],
                                    columns=['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity'])
    predicted_quality = model.predict(banana_features)
    try:
        predicted_quality = float(np.array(predicted_quality).ravel()[0])
    except Exception:
        predicted_quality = predicted_quality[0]
    print(f"\n=== New Prediction ===")
    print(f"Banana specs: {size}, kg of {weight}, level of Sweetness: {sweetness}, level of Softness: {softness}, amount of time {harvesttime}, level of Ripeness: {ripeness}, level of Acidity: {acidity} ")
    print(f"Predicted quality: {predicted_quality:.2f}")
    return predicted_quality

if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

