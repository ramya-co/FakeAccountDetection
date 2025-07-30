import pickle
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import os

from .feature_extractor import FeatureExtractor
from .data_generator import DataGenerator

class FakeAccountDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.get_feature_names()
        self.model_path = 'fake_detection/models/fake_detector_model.pkl'
        self.scaler_path = 'fake_detection/models/scaler.pkl'
        
    def prepare_training_data(self, data_file='data/training_data.csv'):
        """Prepare training data from CSV file"""
        if not os.path.exists(data_file):
            print("Training data not found. Generating new data...")
            generator = DataGenerator()
            generator.generate_training_data(1000, 1000)
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Extract features
        features_list = []
        labels = []
        
        for _, row in df.iterrows():
            user_data = {
                'username': row['username'],
                'bio': row['bio'],
                'created_at': row['created_at'],
                'follower_count': row['follower_count'],
                'following_count': row['following_count'],
                'post_count': row['post_count']
            }
            
            features = self.feature_extractor.extract_features(user_data)
            features_list.append([features.get(feature, 0) for feature in self.feature_names])
            labels.append(1 if row['is_fake'] else 0)
        
        X = np.array(features_list)
        y = np.array(labels)
        
        return X, y
    
    def train_model(self, X, y):
        """Train the fake account detection model"""
        print("Training fake account detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def save_model(self):
        """Save the trained model and scaler"""
        os.makedirs('fake_detection/models', exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Model loaded successfully")
            return True
        else:
            print("No trained model found")
            return False
    
    def predict_single_user(self, user_data):
        """Predict if a single user is fake"""
        if self.model is None:
            if not self.load_model():
                raise Exception("No trained model available")
        
        # Extract features
        features = self.feature_extractor.extract_features(user_data)
        feature_vector = [features.get(feature, 0) for feature in self.feature_names]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0]
        
        return {
            'is_fake': bool(prediction),
            'fake_probability': probability[1],
            'real_probability': probability[0],
            'features': features
        }
    
    def predict_batch(self, users_data):
        """Predict for multiple users"""
        if self.model is None:
            if not self.load_model():
                raise Exception("No trained model available")
        
        results = []
        
        for user_data in users_data:
            try:
                result = self.predict_single_user(user_data)
                result['username'] = user_data.get('username', 'Unknown')
                results.append(result)
            except Exception as e:
                print(f"Error processing user {user_data.get('username', 'Unknown')}: {e}")
                results.append({
                    'username': user_data.get('username', 'Unknown'),
                    'is_fake': False,
                    'fake_probability': 0.0,
                    'real_probability': 1.0,
                    'error': str(e)
                })
        
        return results
    
    def explain_prediction(self, user_data):
        """Explain the prediction using SHAP values"""
        if self.model is None:
            if not self.load_model():
                raise Exception("No trained model available")
        
        # Extract features
        features = self.feature_extractor.extract_features(user_data)
        feature_vector = [features.get(feature, 0) for feature in self.feature_names]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(feature_vector_scaled)
        
        # Get feature importance
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = abs(shap_values[1][0][i])
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'feature_importance': dict(sorted_features[:10]),  # Top 10 features
            'shap_values': shap_values[1][0].tolist(),
            'base_value': explainer.expected_value[1]
        }
    
    def get_feature_importance(self):
        """Get overall feature importance from the model"""
        if self.model is None:
            if not self.load_model():
                raise Exception("No trained model available")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def evaluate_model(self, test_data_file='data/test_data.csv'):
        """Evaluate the model on test data"""
        if not os.path.exists(test_data_file):
            print("Test data not found. Generating test data...")
            generator = DataGenerator()
            test_accounts = generator.generate_test_data(100)
            
            # Save test data
            df = pd.DataFrame(test_accounts)
            df.to_csv(test_data_file, index=False)
        else:
            df = pd.read_csv(test_data_file)
        
        # Prepare test data
        features_list = []
        labels = []
        
        for _, row in df.iterrows():
            user_data = {
                'username': row['username'],
                'bio': row['bio'],
                'created_at': row['created_at'],
                'follower_count': row['follower_count'],
                'following_count': row['following_count'],
                'post_count': row['post_count']
            }
            
            features = self.feature_extractor.extract_features(user_data)
            features_list.append([features.get(feature, 0) for feature in self.feature_names])
            labels.append(1 if row['is_fake'] else 0)
        
        X_test = np.array(features_list)
        y_test = np.array(labels)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist(),
            'true_labels': y_test.tolist()
        }

def main():
    """Main function to train and test the model"""
    detector = FakeAccountDetector()
    
    # Prepare training data
    X, y = detector.prepare_training_data()
    
    # Train model
    accuracy = detector.train_model(X, y)
    
    # Evaluate model
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    detector.evaluate_model()
    
    # Show feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    importance = detector.get_feature_importance()
    for feature, imp in list(importance.items())[:10]:
        print(f"{feature}: {imp:.4f}")

if __name__ == '__main__':
    main() 