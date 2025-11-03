import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import joblib
import os
import base64
from io import BytesIO

class WasteEfficiencyClassifier:
    """Logistic Regression model for classifying waste management efficiency."""
    
    def __init__(self, C=1.0, solver='liblinear', multi_class='auto'):
        """
        Initialize the Logistic Regression classifier.
        
        Parameters:
        -----------
        C : float
            Inverse of regularization strength
        solver : str
            Algorithm to use in the optimization problem
        multi_class : str
            Strategy for multi-class classification
        """
        self.C = C
        self.solver = solver
        self.multi_class = multi_class
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=self.C, 
                solver=self.solver, 
                multi_class=self.multi_class,
                max_iter=1000,
                random_state=42
            ))
        ])
        self.feature_names = None
        self.trained = False
        self.metrics = {}
        self.classes = None
        
    def fit(self, X, y, optimize_hyperparams=False):
        """
        Fit the logistic regression model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
        optimize_hyperparams : bool
            Whether to find the best hyperparameters using grid search
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values.ravel()
            
        # Store unique classes
        self.classes = np.unique(y)
        
        # Find best hyperparameters if requested
        if optimize_hyperparams:
            print("Finding optimal hyperparameters...")
            best_params = self._find_best_params(X, y)
            print(f"Best parameters: {best_params}")
            self.C = best_params['classifier__C']
            self.solver = best_params['classifier__solver']
            self.multi_class = best_params.get('classifier__multi_class', 'auto')
            
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    C=self.C, 
                    solver=self.solver, 
                    multi_class=self.multi_class,
                    max_iter=1000,
                    random_state=42
                ))
            ])
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model
        print("Training logistic regression model...")
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Get probabilities for ROC-AUC calculation
        if len(self.classes) == 2:  # Binary classification
            train_proba = self.model.predict_proba(X_train)[:, 1]
            test_proba = self.model.predict_proba(X_test)[:, 1]
            train_auc = roc_auc_score(y_train, train_proba)
            test_auc = roc_auc_score(y_test, test_proba)
        else:  # Multi-class classification
            train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train), multi_class='ovr')
            test_auc = roc_auc_score(y_test, self.model.predict_proba(X_test), multi_class='ovr')
        
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_precision': precision_score(y_train, train_pred, average='weighted'),
            'test_precision': precision_score(y_test, test_pred, average='weighted'),
            'train_recall': recall_score(y_train, train_pred, average='weighted'),
            'test_recall': recall_score(y_test, test_pred, average='weighted'),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'train_auc': train_auc,
            'test_auc': test_auc,
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }
        
        print(f"Model trained successfully!")
        print(f"Train Accuracy: {self.metrics['train_accuracy']:.4f}, F1: {self.metrics['train_f1']:.4f}")
        print(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}, F1: {self.metrics['test_f1']:.4f}")
        
        self.trained = True
        return self
    
    def _find_best_params(self, X, y):
        """Find the best hyperparameters using grid search."""
        # Define parameter grid
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag']
        }
        
        # For multi-class problems, add multi_class parameter
        if len(np.unique(y)) > 2:
            param_grid['classifier__multi_class'] = ['ovr', 'multinomial']
            # Remove solvers incompatible with multinomial
            param_grid['classifier__solver'] = ['newton-cg', 'lbfgs', 'sag']
        
        # Create a pipeline with the same structure
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_
    
    def predict(self, X):
        """
        Make class predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        model_data = {
            'model': self.model,
            'C': self.C,
            'solver': self.solver,
            'multi_class': self.multi_class,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'trained': self.trained,
            'classes': self.classes
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            C=model_data['C'], 
            solver=model_data['solver'], 
            multi_class=model_data['multi_class']
        )
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']
        instance.trained = model_data['trained']
        instance.classes = model_data['classes']
        
        return instance
    
    def visualize_model_performance(self, X_test=None, y_test=None):
        """
        Create visualizations of model performance.
        
        Parameters:
        -----------
        X_test : array-like or DataFrame, optional
            Test features to use for additional evaluation
        y_test : array-like, optional
            True test labels to use for additional evaluation
            
        Returns:
        --------
        plt_base64 : str
            Base64 encoded PNG image of the performance plot
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        plt.figure(figsize=(15, 10))
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 2. Metrics
        plt.subplot(2, 2, 2)
        plt.axis('off')
        metrics_text = f"Model Performance Metrics:\n\n" \
                      f"Train Accuracy: {self.metrics['train_accuracy']:.4f}\n" \
                      f"Test Accuracy: {self.metrics['test_accuracy']:.4f}\n\n" \
                      f"Train Precision: {self.metrics['train_precision']:.4f}\n" \
                      f"Test Precision: {self.metrics['test_precision']:.4f}\n\n" \
                      f"Train Recall: {self.metrics['train_recall']:.4f}\n" \
                      f"Test Recall: {self.metrics['test_recall']:.4f}\n\n" \
                      f"Train F1: {self.metrics['train_f1']:.4f}\n" \
                      f"Test F1: {self.metrics['test_f1']:.4f}\n\n" \
                      f"Train AUC: {self.metrics['train_auc']:.4f}\n" \
                      f"Test AUC: {self.metrics['test_auc']:.4f}\n"
        plt.text(0.1, 0.5, metrics_text, fontsize=12)
        
        # 3. Class Distribution
        plt.subplot(2, 2, 3)
        class_counts = {}
        for c in self.classes:
            class_counts[c] = sum(1 for label in self.metrics['classification_report'] 
                                 if label.isdigit() and int(label) == c)
        
        class_labels = [f'Class {c}' for c in self.classes]
        sns.barplot(x=class_labels, y=list(class_counts.values()))
        plt.title('Class Distribution')
        plt.xlabel('Efficiency Class')
        plt.ylabel('Count')
        
        # 4. Per-class metrics
        plt.subplot(2, 2, 4)
        class_metrics = []
        
        for c in self.classes:
            if str(c) in self.metrics['classification_report']:
                class_metrics.append({
                    'class': f'Class {c}',
                    'precision': self.metrics['classification_report'][str(c)]['precision'],
                    'recall': self.metrics['classification_report'][str(c)]['recall'],
                    'f1': self.metrics['classification_report'][str(c)]['f1-score']
                })
        
        class_metrics_df = pd.DataFrame(class_metrics)
        class_metrics_melted = pd.melt(class_metrics_df, id_vars=['class'], 
                                       value_vars=['precision', 'recall', 'f1'],
                                       var_name='Metric', value_name='Score')
        
        sns.barplot(x='class', y='Score', hue='Metric', data=class_metrics_melted)
        plt.title('Per-class Performance Metrics')
        plt.xlabel('Efficiency Class')
        plt.ylabel('Score')
        plt.legend(title='Metric')
        
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
            
        # Get the coefficients from the logistic regression model
        classifier = self.model.named_steps['classifier']
        coefficients = classifier.coef_
        
        plt.figure(figsize=(12, 8))
        
        # For binary classification or OvR multiclass
        if coefficients.shape[0] == 1 or self.multi_class == 'ovr':
            # For binary, we use the single coefficient array
            # For OvR, we'll show importance for each class
            
            if len(self.classes) == 2:  # Binary case, single coefficient array
                coefs = coefficients[0]
                plt.subplot(1, 1, 1)
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': np.abs(coefs)
                }).sort_values('Importance', ascending=False)
                
                colors = ['green' if c > 0 else 'red' for c in coefs[feature_importance['Feature'].map(
                    {f: i for i, f in enumerate(self.feature_names)}).values]]
                
                plt.barh(range(len(feature_importance)), feature_importance['Importance'], color=colors)
                plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
                plt.xlabel('Absolute Coefficient Value')
                plt.title(f'Feature Importance for Binary Classification')
                
                # Add a note about interpretation
                plt.figtext(0.5, 0.01, 'Green: Increases probability of Class 1, Red: Decreases probability of Class 1',
                           horizontalalignment='center', fontsize=10)
            
            else:  # Multiclass OvR case, coefficient array for each class
                n_classes = len(self.classes)
                n_features = len(self.feature_names)
                
                for i, class_idx in enumerate(self.classes):
                    plt.subplot(1, n_classes, i+1)
                    
                    class_coefs = coefficients[i]
                    feature_importance = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Importance': np.abs(class_coefs)
                    }).sort_values('Importance', ascending=False).head(min(10, n_features))
                    
                    colors = ['green' if c > 0 else 'red' for c in class_coefs[feature_importance['Feature'].map(
                        {f: i for i, f in enumerate(self.feature_names)}).values]]
                    
                    plt.barh(range(len(feature_importance)), feature_importance['Importance'], color=colors)
                    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
                    plt.xlabel('Absolute Coefficient Value')
                    plt.title(f'Feature Importance for Class {class_idx}')
                    
                # Add a note about interpretation
                plt.figtext(0.5, 0.01, 'Green: Increases probability of this class, Red: Decreases probability',
                           horizontalalignment='center', fontsize=10)
        
        else:  # Multinomial case - coefficients are less directly interpretable
            n_classes = len(self.classes)
            n_features = len(self.feature_names)
            
            # Sum absolute coefficient values across all classes for each feature
            feature_importance = np.sum(np.abs(coefficients), axis=0)
            
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            plt.barh(range(min(20, n_features)), feature_importance_df['Importance'].head(20), color='skyblue')
            plt.yticks(range(min(20, n_features)), feature_importance_df['Feature'].head(20))
            plt.xlabel('Sum of Absolute Coefficient Values Across Classes')
            plt.title(f'Overall Feature Importance for Multinomial Model')
            
            # Add a note about interpretation
            plt.figtext(0.5, 0.01, 'For multinomial models, importance represents overall impact across all classes',
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
                             'data', 'lr_waste_dataset.csv')
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Split features and target
        X = data.drop('efficiency_class', axis=1)
        y = data['efficiency_class']
        
        # Create and train the model
        model = WasteEfficiencyClassifier()
        model.fit(X, y, optimize_hyperparams=True)
        
        # Save the model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
        model.save_model(model_path)
        
        # Visualize model performance
        viz = model.visualize_model_performance()
        
        # Save visualization
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'webapp', 'static', 'images')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        with open(os.path.join(viz_dir, 'lr_performance.png'), 'wb') as f:
            f.write(base64.b64decode(viz))
        
        # Visualize feature importance
        viz = model.visualize_feature_importance()
        
        with open(os.path.join(viz_dir, 'lr_feature_importance.png'), 'wb') as f:
            f.write(base64.b64decode(viz))
        
        print("Logistic regression model training and visualization complete!")
    else:
        print(f"Data file not found: {data_path}")
        print("Please generate the dataset first by running the generate_dataset.py script.") 