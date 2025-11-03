import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_waste_data(n_samples=1000, output_file='waste_dataset.csv'):
    """
    Generate a synthetic dataset for waste management with features that affect waste generation.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points to generate
    output_file : str
        Path to save the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    print(f"Generating synthetic waste dataset with {n_samples} samples...")
    
    # Generate dates for a year
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i % 365) for i in range(n_samples)]
    
    # Create neighborhood IDs (100 different neighborhoods)
    neighborhood_ids = np.random.randint(1, 101, size=n_samples)
    
    # Generate features
    population_density = np.random.normal(500, 200, n_samples)  # people per sq km
    population_density = np.clip(population_density, 50, 1200)  # clip to reasonable values
    
    income_level = np.random.normal(50000, 20000, n_samples)  # annual income in $
    income_level = np.clip(income_level, 20000, 120000)  # clip to reasonable values
    
    # Base recycling rate correlated with income level
    recycling_rate_base = 0.1 + 0.4 * (income_level - 20000) / 100000  # 10-50% range
    noise = np.random.normal(0, 0.05, n_samples)  # add some noise
    recycling_rate = recycling_rate_base + noise
    recycling_rate = np.clip(recycling_rate, 0.05, 0.85)  # clip to reasonable values
    
    # Public awareness score (1-10)
    public_awareness_base = 3 + 6 * (recycling_rate - 0.05) / 0.8  # correlated with recycling rate
    public_awareness = public_awareness_base + np.random.normal(0, 0.5, n_samples)
    public_awareness = np.clip(public_awareness, 1, 10)
    
    # Commercial activity level (1-10)
    commercial_activity = np.random.normal(5, 2, n_samples)
    commercial_activity = np.clip(commercial_activity, 1, 10)
    
    # Weather temperature in Celsius
    # Generate seasonal pattern with noise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    season_factor = np.sin((day_of_year - 15) / 365 * 2 * np.pi)  # peak in summer, low in winter
    weather_temperature = 15 + 15 * season_factor + np.random.normal(0, 3, n_samples)
    
    # Is holiday flag
    is_holiday = np.zeros(n_samples, dtype=int)
    holiday_dates = [
        "01-01", "01-15", "02-14", "02-19", "03-17", "04-01", "05-05", "05-29", 
        "06-19", "07-04", "09-04", "10-31", "11-23", "11-24", "12-24", "12-25", "12-31"
    ]
    
    for i, date in enumerate(dates):
        if date.strftime("%m-%d") in holiday_dates:
            is_holiday[i] = 1
    
    # Is weekend flag
    is_weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates])
    
    # === TARGET VARIABLES ===
    
    # 1. Waste generation (kg per person per day)
    # Base waste depends on income, commercial activity, and population density
    waste_base = (
        0.8  # base value
        + 0.5 * (income_level - 20000) / 100000  # income effect (0-0.5 kg)
        + 0.3 * (commercial_activity - 1) / 9  # commercial effect (0-0.3 kg)
        - 0.2 * (recycling_rate - 0.05) / 0.8  # recycling effect (reduces waste by 0-0.2 kg)
    )
    
    # Add time-based patterns
    # Weekends tend to have more waste
    weekend_effect = 0.2 * is_weekend
    
    # Holidays tend to have more waste
    holiday_effect = 0.5 * is_holiday
    
    # Seasonal effects (more in summer, less in winter)
    seasonal_effect = 0.1 * season_factor
    
    # Temperature effect (more waste in warmer weather)
    temperature_effect = 0.1 * (weather_temperature - 0) / 30
    
    # Add specific non-linear patterns and noise
    nonlinear_factor = 0.05 * np.sin(population_density / 100) + 0.05 * np.cos(weather_temperature / 10)
    noise = np.random.normal(0, 0.1, n_samples)
    
    # Combine all factors to calculate waste generation
    waste_generation = (
        waste_base 
        + weekend_effect 
        + holiday_effect 
        + seasonal_effect 
        + temperature_effect 
        + nonlinear_factor 
        + noise
    )
    waste_generation = np.clip(waste_generation, 0.5, 3.0)  # clip to reasonable values (kg per person per day)
    
    # Scale by population density to get total waste for the area (tons per sq km per day)
    total_waste = waste_generation * population_density / 1000  # Convert to tons
    
    # 2. Efficiency class (0: Very Efficient, 1: Efficient, 2: Moderate, 3: Inefficient)
    # Calculated based on waste generation relative to similar neighborhoods
    
    # Create a normalized waste score based on income and commercial activity
    normalized_waste = waste_generation / (0.8 + 0.5 * (income_level - 20000) / 100000 + 0.3 * (commercial_activity - 1) / 9)
    
    # Set thresholds for classification
    efficiency_class = np.zeros(n_samples, dtype=int)
    efficiency_class[normalized_waste > 0.9] = 1  # Efficient
    efficiency_class[normalized_waste > 1.0] = 2  # Moderate
    efficiency_class[normalized_waste > 1.1] = 3  # Inefficient
    
    # Create the DataFrame
    data = pd.DataFrame({
        'date': dates,
        'neighborhood_id': neighborhood_ids,
        'population_density': population_density,
        'income_level': income_level,
        'recycling_rate': recycling_rate,
        'public_awareness': public_awareness,
        'commercial_activity': commercial_activity,
        'weather_temperature': weather_temperature,
        'is_holiday': is_holiday,
        'is_weekend': is_weekend,
        'waste_generation': waste_generation,
        'total_waste': total_waste,
        'efficiency_class': efficiency_class
    })
    
    # Save the dataset to CSV
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    data.to_csv(data_path, index=False)
    print(f"Dataset saved to {data_path}")
    
    # Also save a version with specific features for each model
    # 1. For Polynomial Regression - time-based waste patterns
    pr_data = prepare_polynomial_regression_data(data)
    pr_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pr_' + output_file)
    pr_data.to_csv(pr_data_path, index=False)
    
    # 2. For Logistic Regression - efficiency classification
    lr_data = prepare_logistic_regression_data(data)
    lr_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lr_' + output_file)
    lr_data.to_csv(lr_data_path, index=False)
    
    # 3. For KNN Clustering - neighborhood profiles
    knn_data = prepare_knn_clustering_data(data)
    knn_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knn_' + output_file)
    knn_data.to_csv(knn_data_path, index=False)
    
    return data

def prepare_polynomial_regression_data(data):
    """Prepare data for polynomial regression by including relevant features."""
    # For polynomial regression, we focus on time-based patterns
    # Extract features like day of week, day of month, month, etc.
    
    pr_data = data.copy()
    
    # Add engineered features
    pr_data['day_of_week'] = pr_data['date'].dt.dayofweek
    pr_data['day_of_month'] = pr_data['date'].dt.day
    pr_data['month'] = pr_data['date'].dt.month
    pr_data['quarter'] = pr_data['date'].dt.quarter
    
    # Create sin and cos features for cyclical patterns
    pr_data['day_of_year_sin'] = np.sin(2 * np.pi * pr_data['date'].dt.dayofyear / 365)
    pr_data['day_of_year_cos'] = np.cos(2 * np.pi * pr_data['date'].dt.dayofyear / 365)
    
    # Features to include for polynomial regression
    features = [
        'population_density', 'income_level', 'commercial_activity', 
        'weather_temperature', 'is_holiday', 'is_weekend',
        'day_of_week', 'day_of_month', 'month', 'day_of_year_sin', 'day_of_year_cos'
    ]
    
    target = ['waste_generation']
    
    return pr_data[features + target]

def prepare_logistic_regression_data(data):
    """Prepare data for logistic regression to classify efficiency."""
    
    # For efficiency classification, we use socioeconomic and behavioral features
    features = [
        'population_density', 'income_level', 'recycling_rate', 
        'public_awareness', 'commercial_activity', 'waste_generation'
    ]
    
    target = ['efficiency_class']
    
    return data[features + target]

def prepare_knn_clustering_data(data):
    """Prepare data for KNN clustering to identify similar neighborhoods."""
    
    # For clustering, we focus on the overall waste profile of neighborhoods
    # Group by neighborhood_id and compute aggregated statistics
    
    neighborhood_profiles = data.groupby('neighborhood_id').agg({
        'population_density': 'mean',
        'income_level': 'mean',
        'recycling_rate': 'mean',
        'public_awareness': 'mean',
        'commercial_activity': 'mean',
        'waste_generation': ['mean', 'std'],
        'total_waste': 'mean',
        'efficiency_class': lambda x: x.mode()[0]  # most common efficiency class
    })
    
    # Flatten the multi-index columns
    neighborhood_profiles.columns = ['_'.join(col).strip('_') for col in neighborhood_profiles.columns.values]
    
    # Reset index to make neighborhood_id a column again
    neighborhood_profiles = neighborhood_profiles.reset_index()
    
    return neighborhood_profiles

def visualize_dataset(data, output_dir=None):
    """Create visualizations of the generated dataset."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Creating visualizations...")
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Waste generation distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['waste_generation'], kde=True)
    plt.title('Distribution of Waste Generation (kg per person per day)')
    plt.xlabel('Waste Generation')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'waste_distribution.png'))
    plt.close()
    
    # 2. Waste generation by day of week
    plt.figure(figsize=(10, 6))
    data['day_of_week'] = data['date'].dt.dayofweek
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_waste = data.groupby('day_of_week')['waste_generation'].mean().reindex(range(7))
    sns.barplot(x=day_names, y=daily_waste)
    plt.title('Average Waste Generation by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Waste Generation (kg per person)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waste_by_day.png'))
    plt.close()
    
    # 3. Waste generation over the year (seasonal patterns)
    plt.figure(figsize=(12, 6))
    data['month'] = data['date'].dt.month
    monthly_waste = data.groupby('month')['waste_generation'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.lineplot(x=month_names, y=monthly_waste)
    plt.title('Seasonal Pattern of Waste Generation')
    plt.xlabel('Month')
    plt.ylabel('Average Waste Generation (kg per person)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'seasonal_waste.png'))
    plt.close()
    
    # 4. Impact of holidays on waste generation
    plt.figure(figsize=(10, 6))
    holiday_waste = data.groupby('is_holiday')['waste_generation'].mean()
    sns.barplot(x=['Regular Day', 'Holiday'], y=holiday_waste)
    plt.title('Waste Generation: Holidays vs. Regular Days')
    plt.xlabel('Day Type')
    plt.ylabel('Average Waste Generation (kg per person)')
    plt.savefig(os.path.join(output_dir, 'holiday_waste.png'))
    plt.close()
    
    # 5. Correlation between income level and waste generation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='income_level', y='waste_generation', data=data)
    plt.title('Relationship between Income Level and Waste Generation')
    plt.xlabel('Income Level ($)')
    plt.ylabel('Waste Generation (kg per person)')
    plt.savefig(os.path.join(output_dir, 'income_waste.png'))
    plt.close()
    
    # 6. Distribution of efficiency classes
    plt.figure(figsize=(10, 6))
    class_names = ['Very Efficient', 'Efficient', 'Moderate', 'Inefficient']
    class_counts = data['efficiency_class'].value_counts().sort_index()
    sns.barplot(x=class_names, y=class_counts)
    plt.title('Distribution of Waste Efficiency Classes')
    plt.xlabel('Efficiency Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'efficiency_classes.png'))
    plt.close()
    
    # 7. Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    features = ['population_density', 'income_level', 'recycling_rate', 
                'public_awareness', 'commercial_activity', 'weather_temperature',
                'is_holiday', 'is_weekend', 'waste_generation', 'efficiency_class']
    corr = data[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Generate the dataset with 2000 samples
    data = generate_synthetic_waste_data(n_samples=2000, output_file='waste_dataset.csv')
    
    # Create visualizations
    visualize_dataset(data)
    
    print("Dataset generation complete!") 