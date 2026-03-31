# ===============================
# SALES FORECASTING USING ML
# ===============================

"""
Professional Sales Forecasting Script

This script performs sales forecasting using machine learning on Superstore dataset.
It includes data loading, preprocessing, model training, evaluation, and visualization.
"""

# Import Required Libraries
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # More robust model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = r"C:\Users\pavit\OneDrive\Documents\Future Intern\task_1\Sample - Superstore.csv"
OUTPUT_DIR = "output"
FIG_WIDTH = 8
FIG_HEIGHT = 6
FORECAST_DAYS = 30

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and aggregate sales data."""
    # Convert Order Date to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values found: {missing[missing > 0]}")

    # Aggregate Sales by Date
    sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()

    # Create Time-Based Features
    sales_data['Year'] = sales_data['Order Date'].dt.year
    sales_data['Month'] = sales_data['Order Date'].dt.month
    sales_data['Day'] = sales_data['Order Date'].dt.day
    sales_data['DayOfWeek'] = sales_data['Order Date'].dt.dayofweek
    sales_data['Quarter'] = sales_data['Order Date'].dt.quarter

    logger.info(f"Preprocessed data: {len(sales_data)} daily sales records")
    return sales_data

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Train the forecasting model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Model trained successfully")
    return model

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    metrics = {
        'MAE': mae,
        'RMSE': rmse
    }

    logger.info(f"Model Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return metrics, predictions

def forecast_future(model: RandomForestRegressor, sales_data: pd.DataFrame, days: int) -> pd.DataFrame:
    """Generate future sales forecast."""
    last_date = sales_data['Order Date'].max()
    future_dates = pd.date_range(start=last_date, periods=days + 1)[1:]

    future_df = pd.DataFrame({'Order Date': future_dates})
    future_df['Year'] = future_df['Order Date'].dt.year
    future_df['Month'] = future_df['Order Date'].dt.month
    future_df['Day'] = future_df['Order Date'].dt.day
    future_df['DayOfWeek'] = future_df['Order Date'].dt.dayofweek
    future_df['Quarter'] = future_df['Order Date'].dt.quarter

    future_predictions = model.predict(future_df[['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']])

    forecast_df = future_df[['Order Date']].copy()
    forecast_df['Predicted Sales'] = future_predictions

    logger.info(f"Generated {days}-day forecast")
    return forecast_df

def plot_historical_sales(sales_data: pd.DataFrame, output_dir: str):
    """Plot and save historical sales trend."""
    sales_data['Sales_Rolling7'] = sales_data['Sales'].rolling(window=7, min_periods=1).mean()

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.bar(sales_data['Order Date'], sales_data['Sales'], width=2, alpha=0.4, label='Daily Sales')
    plt.plot(sales_data['Order Date'], sales_data['Sales_Rolling7'], color='tab:red', label='7-day MA')
    plt.title("Sales Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'historical_sales.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved historical sales plot")

def plot_actual_vs_predicted(test_dates: pd.Series, y_test: pd.Series, predictions: np.ndarray, output_dir: str):
    """Plot and save actual vs predicted sales."""
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(test_dates, y_test, label='Actual Sales', color='tab:green', linewidth=2, marker='o', markersize=4)
    plt.plot(test_dates, predictions, label='Predicted Sales', color='tab:orange', linewidth=2, marker='s', markersize=4)
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved actual vs predicted plot")

def plot_forecast(sales_data: pd.DataFrame, forecast_df: pd.DataFrame, output_dir: str):
    """Plot and save future sales forecast."""
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(sales_data['Order Date'], sales_data['Sales'], label='Historical Sales', color='tab:blue', alpha=0.4)
    plt.bar(forecast_df['Order Date'], forecast_df['Predicted Sales'], width=2, alpha=0.6, label='Forecast Sales', color='tab:red')
    plt.title("Future Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_forecast.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved forecast plot")

def save_forecast_table(forecast_df: pd.DataFrame, output_dir: str):
    """Save forecast results to CSV."""
    forecast_df.to_csv(os.path.join(output_dir, 'forecast_results.csv'), index=False)
    logger.info("Saved forecast table to CSV")

def main():
    """Main execution function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and preprocess data
    df = load_data(DATA_PATH)
    sales_data = preprocess_data(df)

    # Prepare features
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']
    X = sales_data[features]
    y = sales_data['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    metrics, predictions = evaluate_model(model, X_test, y_test)

    # Generate forecast
    forecast_df = forecast_future(model, sales_data, FORECAST_DAYS)

    # Create plots
    test_dates = sales_data['Order Date'].iloc[len(X_train):].reset_index(drop=True)
    plot_historical_sales(sales_data, OUTPUT_DIR)
    plot_actual_vs_predicted(test_dates, y_test, predictions, OUTPUT_DIR)
    plot_forecast(sales_data, forecast_df, OUTPUT_DIR)

    # Save forecast table
    save_forecast_table(forecast_df, OUTPUT_DIR)

    # Print summary
    print("\n" + "="*50)
    print("SALES FORECASTING SUMMARY")
    print("="*50)
    print(f"Model: Random Forest Regressor")
    print(f"Training Data: {len(X_train)} samples")
    print(f"Test Data: {len(X_test)} samples")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"Forecast Period: {FORECAST_DAYS} days")
    print(f"Output saved to: {OUTPUT_DIR}/")
    print("="*50)

if __name__ == "__main__":
    main()