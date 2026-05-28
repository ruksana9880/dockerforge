import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class CardiovascularDiseasePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, filepath='cardiovascular_dataset.csv'):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)
        
        # Separate features and target
        self.X = self.df.drop('cardiovascular_disease', axis=1)
        self.y = self.df['cardiovascular_disease']
        self.feature_names = self.X.columns.tolist()
        
        print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Positive class ratio: {self.y.mean():.2%}")
        
        return self.X, self.y
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Split and scale the data"""
        print("Preprocessing data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def train_models(self):
        """Train multiple models and compare performance"""
        print("Training multiple models...")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        self.model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.model_scores[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
        
        # Select best model based on F1-score
        best_model_name = max(self.model_scores.keys(), 
                             key=lambda x: self.model_scores[x]['f1'])
        self.best_model = self.model_scores[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print(f"\nPerforming hyperparameter tuning on {self.best_model_name}...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            X_train_data = self.X_train
            X_test_data = self.X_test
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            X_train_data = self.X_train
            X_test_data = self.X_test
            
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
            X_train_data = self.X_train_scaled
            X_test_data = self.X_test_scaled
            
        else:  # SVM
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            model = SVC(probability=True, random_state=42)
            X_train_data = self.X_train_scaled
            X_test_data = self.X_test_scaled
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_data, self.y_train)
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        # Evaluate tuned model
        y_pred = self.best_model.predict(X_test_data)
        y_pred_proba = self.best_model.predict_proba(X_test_data)[:, 1]
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Test F1-score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"Test AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        print("\nAnalyzing feature importance...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("Feature importance not available for this model.")
            return
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            y_pred = self.best_model.predict(self.X_test_scaled)
        else:
            y_pred = self.best_model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No CVD', 'CVD'], 
                   yticklabels=['No CVD', 'CVD'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the trained model and scaler"""
        print("\nSaving model and scaler...")
        
        # Save model
        joblib.dump(self.best_model, 'cvd_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'cvd_scaler.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'feature_names.pkl')
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'model_type': type(self.best_model).__name__
        }
        joblib.dump(model_info, 'model_info.pkl')
        
        print("Model saved successfully!")
        print("Files saved:")
        print("- cvd_model.pkl")
        print("- cvd_scaler.pkl")
        print("- feature_names.pkl")
        print("- model_info.pkl")
    
    def generate_model_report(self):
        """Generate a comprehensive model report"""
        print("\n" + "="*50)
        print("CARDIOVASCULAR DISEASE PREDICTION MODEL REPORT")
        print("="*50)
        
        print(f"\nDataset Information:")
        print(f"- Total samples: {len(self.df)}")
        print(f"- Features: {len(self.feature_names)}")
        print(f"- Positive cases: {self.y.sum()} ({self.y.mean():.2%})")
        print(f"- Training samples: {len(self.y_train)}")
        print(f"- Test samples: {len(self.y_test)}")
        
        print(f"\nBest Model: {self.best_model_name}")
        
        # Get predictions for best model
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            y_pred = self.best_model.predict(self.X_test_scaled)
            y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        print(f"\nModel Performance:")
        print(f"- Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"- Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"- Recall: {recall_score(self.y_test, y_pred):.4f}")
        print(f"- F1-Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"- AUC-ROC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['No CVD', 'CVD']))

def main():
    # Initialize predictor
    predictor = CardiovascularDiseasePredictor()
    
    # Load and preprocess data
    predictor.load_data()
    predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Perform hyperparameter tuning
    predictor.hyperparameter_tuning()
    
    # Analyze results
    predictor.analyze_feature_importance()
    predictor.plot_confusion_matrix()
    predictor.plot_roc_curve()
    
    # Save model
    predictor.save_model()
    
    # Generate report
    predictor.generate_model_report()

if __name__ == "__main__":
    main()