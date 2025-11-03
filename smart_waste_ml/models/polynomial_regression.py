import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import base64
from io import BytesIO

class WastePolynomialRegression:
    """Polynomial Regression model for predicting waste generation based on temporal patterns."""
    
    def __init__(self, degree=3):
        """
        Initialize the Polynomial Regression model.
        
        Parameters:
        -----------
        degree : int
            Degree of the polynomial features
        """
        self.degree = degree
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        self.feature_names = None
        self.trained = False
        self.metrics = {}
        
    def fit(self, X, y, find_best_degree=False, max_degree=5):
        """
        Fit the polynomial regression model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
        find_best_degree : bool
            Whether to find the best polynomial degree using cross-validation
        max_degree : int
            Maximum degree to consider when find_best_degree is True
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values.ravel()
            
        # Find best degree if requested
        if find_best_degree:
            print("Finding optimal polynomial degree...")
            best_degree = self._find_best_degree(X, y, max_degree)
            print(f"Best degree: {best_degree}")
            self.degree = best_degree
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model
        print("Training polynomial regression model...")
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print(f"Model trained successfully!")
        print(f"Train RMSE: {self.metrics['train_rmse']:.4f}, R²: {self.metrics['train_r2']:.4f}")
        print(f"Test RMSE: {self.metrics['test_rmse']:.4f}, R²: {self.metrics['test_r2']:.4f}")
        
        self.trained = True
        return self
    
    def _find_best_degree(self, X, y, max_degree):
        """Find the best polynomial degree using cross-validation."""
        degrees = list(range(1, max_degree + 1))
        best_score = -np.inf
        best_degree = 1
        
        for degree in degrees:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            
            # Perform k-fold cross-validation
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            mean_score = np.mean(scores)
            
            print(f"Degree {degree}: Mean R² = {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_degree = degree
        
        return best_degree
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        model_data = {
            'model': self.model,
            'degree': self.degree,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'trained': self.trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model_data = joblib.load(filepath)
        
        instance = cls(degree=model_data['degree'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']
        instance.trained = model_data['trained']
        
        return instance
    
    def visualize_model_performance(self, X, y, num_samples=200):
        """
        Create a visualization of model performance.
        
        Parameters:
        -----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,)
            True target values
        num_samples : int
            Number of samples to plot (randomly selected)
            
        Returns:
        --------
        plt_base64 : str
            Base64 encoded PNG image of the plot
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values.ravel()
            
        # Make predictions
        y_pred = self.predict(X)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # If there are too many points, sample a subset
        if len(y) > num_samples:
            indices = np.random.choice(len(y), num_samples, replace=False)
            y_sample = y[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_sample = y
            y_pred_sample = y_pred
            
        # Plot actual vs predicted values
        plt.subplot(2, 2, 1)
        plt.scatter(y_sample, y_pred_sample, alpha=0.5)
        plt.plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 'r--')
        plt.xlabel('Actual Waste Generation')
        plt.ylabel('Predicted Waste Generation')
        plt.title('Actual vs Predicted Values')
        
        # Plot residuals
        residuals = y_pred - y
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Plot residual distribution
        plt.subplot(2, 2, 3)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        # Plot metrics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics_text = f"Model Performance Metrics:\n\n" \
                      f"Training RMSE: {self.metrics['train_rmse']:.4f}\n" \
                      f"Test RMSE: {self.metrics['test_rmse']:.4f}\n" \
                      f"Training R²: {self.metrics['train_r2']:.4f}\n" \
                      f"Test R²: {self.metrics['test_r2']:.4f}\n\n" \
                      f"Polynomial Degree: {self.degree}"
        plt.text(0.1, 0.5, metrics_text, fontsize=12)
        
        plt.tight_layout()
        
        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert PNG to base64 string
        plt_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plt.close()
        
        return plt_base64
    
    def visualize_feature_importance(self):
        """
        Visualize feature importance for the model.
        
        Returns:
        --------
        plt_base64 : str
            Base64 encoded PNG image of the plot
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        # Get the coefficients from the linear regression part
        linear_model = self.model.named_steps['linear']
        coefficients = linear_model.coef_
        
        # Get the feature names from polynomial features
        poly = self.model.named_steps['poly']
        if self.feature_names:
            # Use get_feature_names_out if scikit-learn version >= 1.0
            try:
                poly_feature_names = poly.get_feature_names_out(self.feature_names)
            except AttributeError:
                # For older scikit-learn versions
                poly_feature_names = poly.get_feature_names(self.feature_names)
        else:
            # If no feature names were provided, use generic names
            try:
                poly_feature_names = poly.get_feature_names_out()
            except AttributeError:
                poly_feature_names = poly.get_feature_names()
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': poly_feature_names,
            'Coefficient': coefficients
        })
        
        # Sort by absolute coefficient value
        importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient'])
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
        
        # Plot top 20 features (or all if less than 20)
        n_features = min(20, len(importance_df))
        top_features = importance_df.head(n_features)
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
        
        plt.barh(range(n_features), top_features['Coefficient'], color=colors)
        plt.yticks(range(n_features), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {n_features} Feature Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add a note about interpretation
        plt.figtext(0.5, 0.01, 'Green: Increases waste generation, Red: Decreases waste generation',
                   horizontalalignment='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert PNG to base64 string
        plt_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plt.close()
        
        return plt_base64


# Example usage
if __name__ == "__main__":
    # Load the data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data', 'pr_waste_dataset.csv')
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Split features and target
        X = data.drop('waste_generation', axis=1)
        y = data['waste_generation']
        
        # Create and train the model
        model = WastePolynomialRegression(degree=3)
        model.fit(X, y, find_best_degree=True, max_degree=4)
        
        # Save the model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'polynomial_regression_model.joblib')
        model.save_model(model_path)
        
        # Visualize model performance
        viz = model.visualize_model_performance(X, y)
        
        # Save visualization
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'webapp', 'static', 'images')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        with open(os.path.join(viz_dir, 'pr_performance.png'), 'wb') as f:
            f.write(base64.b64decode(viz))
        
        print("Polynomial regression model training and visualization complete!")
    else:
        print(f"Data file not found: {data_path}")
        print("Please generate the dataset first by running the generate_dataset.py script.") 