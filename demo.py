#!/usr/bin/env python3
"""
Demo script for the Fake Account Detection System
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_demo_banner():
    """Print demo banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🔍 FAKE ACCOUNT DETECTION DEMO 🔍                    ║
    ║                                                              ║
    ║  Demonstrating the capabilities of our ML detection system  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def demo_data_generation():
    """Demonstrate data generation"""
    print("📊 DEMO: Data Generation")
    print("-" * 50)
    
    try:
        from fake_detection.data_generator import DataGenerator
        
        generator = DataGenerator()
        
        # Generate sample accounts
        print("Generating sample fake accounts...")
        fake_accounts = generator.generate_fake_accounts(5)
        
        print("Generating sample real accounts...")
        real_accounts = generator.generate_real_accounts(5)
        
        print("\n📋 Sample Fake Accounts:")
        for i, account in enumerate(fake_accounts[:3], 1):
            print(f"  {i}. Username: {account['username']}")
            print(f"     Bio: {account['bio'][:50]}...")
            print(f"     Followers: {account['follower_count']}, Following: {account['following_count']}")
            print(f"     Fake Score: {account['fake_score']:.2f}")
            print()
        
        print("📋 Sample Real Accounts:")
        for i, account in enumerate(real_accounts[:3], 1):
            print(f"  {i}. Username: {account['username']}")
            print(f"     Bio: {account['bio'][:50]}...")
            print(f"     Followers: {account['follower_count']}, Following: {account['following_count']}")
            print(f"     Fake Score: {account['fake_score']:.2f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data generation demo: {e}")
        return False

def demo_feature_extraction():
    """Demonstrate feature extraction"""
    print("🔍 DEMO: Feature Extraction")
    print("-" * 50)
    
    try:
        from fake_detection.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Test cases
        test_cases = [
            {
                'name': 'Suspicious Account',
                'data': {
                    'username': 'user123456',
                    'bio': '💰 Make money fast! Click here: http://scam.com',
                    'created_at': datetime.now().isoformat(),
                    'follower_count': 5,
                    'following_count': 1000,
                    'post_count': 2
                }
            },
            {
                'name': 'Real Account',
                'data': {
                    'username': 'john_doe',
                    'bio': 'Hi, I\'m John! Love photography and travel ✈️',
                    'created_at': (datetime.now() - pd.Timedelta(days=365)).isoformat(),
                    'follower_count': 250,
                    'following_count': 180,
                    'post_count': 45
                }
            }
        ]
        
        for case in test_cases:
            print(f"\n📋 {case['name']}:")
            print(f"  Username: {case['data']['username']}")
            print(f"  Bio: {case['data']['bio']}")
            
            features = extractor.extract_features(case['data'])
            
            print("  🔍 Key Features:")
            print(f"    - Username length: {features.get('username_length', 0)}")
            print(f"    - Username entropy: {features.get('username_entropy', 0):.2f}")
            print(f"    - Bio sentiment: {features.get('bio_sentiment', 0):.2f}")
            print(f"    - Bio suspicious keywords: {features.get('bio_has_suspicious_keywords', 0)}")
            print(f"    - Follower ratio: {features.get('follower_following_ratio', 0):.2f}")
            print(f"    - Account age: {features.get('account_age_days', 0)} days")
            print(f"    - High following/low followers: {features.get('high_following_low_followers', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature extraction demo: {e}")
        return False

def demo_model_prediction():
    """Demonstrate model prediction"""
    print("🤖 DEMO: Model Prediction")
    print("-" * 50)
    
    try:
        from fake_detection.detector import FakeAccountDetector
        from fake_detection.data_generator import DataGenerator
        
        # Generate training data and train model
        print("Training model with sample data...")
        generator = DataGenerator()
        fake_accounts = generator.generate_fake_accounts(100)
        real_accounts = generator.generate_real_accounts(100)
        
        # Create training data
        all_accounts = fake_accounts + real_accounts
        df = pd.DataFrame(all_accounts)
        df.to_csv('data/demo_training_data.csv', index=False)
        
        # Train model
        detector = FakeAccountDetector()
        X, y = detector.prepare_training_data('data/demo_training_data.csv')
        detector.train_model(X, y)
        
        # Test cases
        test_cases = [
            {
                'name': 'Obvious Fake',
                'data': {
                    'username': 'money_maker_2024',
                    'bio': '💸 Earn $1000 daily! Click here: http://getrich.com',
                    'created_at': datetime.now().isoformat(),
                    'follower_count': 2,
                    'following_count': 1500,
                    'post_count': 1
                }
            },
            {
                'name': 'Suspicious Account',
                'data': {
                    'username': 'user987654',
                    'bio': '🔥 Limited time offer! Buy now!',
                    'created_at': (datetime.now() - pd.Timedelta(days=5)).isoformat(),
                    'follower_count': 10,
                    'following_count': 800,
                    'post_count': 5
                }
            },
            {
                'name': 'Real Account',
                'data': {
                    'username': 'sarah_wilson',
                    'bio': 'Photography enthusiast 📸 | Coffee lover ☕ | Travel addict ✈️',
                    'created_at': (datetime.now() - pd.Timedelta(days=500)).isoformat(),
                    'follower_count': 320,
                    'following_count': 280,
                    'post_count': 67
                }
            }
        ]
        
        print("\n🔍 Analysis Results:")
        for case in test_cases:
            result = detector.predict_single_user(case['data'])
            
            print(f"\n📋 {case['name']}:")
            print(f"  Username: {case['data']['username']}")
            
            if result['is_fake']:
                print(f"  🚨 Result: FAKE ACCOUNT")
            else:
                print(f"  ✅ Result: REAL ACCOUNT")
            
            print(f"  📊 Fake Probability: {result['fake_probability']:.2%}")
            print(f"  📊 Real Probability: {result['real_probability']:.2%}")
            
            # Show key features
            features = result['features']
            print(f"  🔍 Key Indicators:")
            print(f"    - Username entropy: {features.get('username_entropy', 0):.2f}")
            print(f"    - Bio suspicious keywords: {features.get('bio_has_suspicious_keywords', 0)}")
            print(f"    - Follower ratio: {features.get('follower_following_ratio', 0):.2f}")
            print(f"    - Account age: {features.get('account_age_days', 0)} days")
        
        # Cleanup
        if os.path.exists('data/demo_training_data.csv'):
            os.remove('data/demo_training_data.csv')
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model prediction demo: {e}")
        return False

def demo_explainability():
    """Demonstrate model explainability"""
    print("🔍 DEMO: Model Explainability")
    print("-" * 50)
    
    try:
        from fake_detection.detector import FakeAccountDetector
        from fake_detection.data_generator import DataGenerator
        
        # Generate training data and train model
        generator = DataGenerator()
        fake_accounts = generator.generate_fake_accounts(50)
        real_accounts = generator.generate_real_accounts(50)
        
        all_accounts = fake_accounts + real_accounts
        df = pd.DataFrame(all_accounts)
        df.to_csv('data/demo_explainability_data.csv', index=False)
        
        detector = FakeAccountDetector()
        X, y = detector.prepare_training_data('data/demo_explainability_data.csv')
        detector.train_model(X, y)
        
        # Test case
        test_user = {
            'username': 'crypto_investor_2024',
            'bio': '🚀 Join our crypto investment program! Earn 500% returns! Click: http://crypto-scam.com',
            'created_at': (datetime.now() - pd.Timedelta(days=3)).isoformat(),
            'follower_count': 5,
            'following_count': 1200,
            'post_count': 2
        }
        
        result = detector.predict_single_user(test_user)
        explanation = detector.explain_prediction(test_user)
        
        print(f"📋 Test Account: {test_user['username']}")
        print(f"🚨 Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
        print(f"📊 Confidence: {result['fake_probability']:.2%}")
        
        print("\n🔍 Top Contributing Features:")
        importance = explanation['feature_importance']
        for i, (feature, importance_value) in enumerate(list(importance.items())[:5], 1):
            print(f"  {i}. {feature}: {importance_value:.4f}")
        
        print("\n💡 Why this account was flagged:")
        features = result['features']
        
        if features.get('bio_has_suspicious_keywords', 0) > 0:
            print("  • Bio contains suspicious keywords (money-making schemes)")
        
        if features.get('high_following_low_followers', 0) == 1:
            print("  • High following count with very few followers")
        
        if features.get('account_age_days', 0) < 30:
            print("  • Very recent account creation")
        
        if features.get('username_has_numbers', 0) == 1:
            print("  • Username contains random numbers")
        
        # Cleanup
        if os.path.exists('data/demo_explainability_data.csv'):
            os.remove('data/demo_explainability_data.csv')
        
        return True
        
    except Exception as e:
        print(f"❌ Error in explainability demo: {e}")
        return False

def main():
    """Run the complete demo"""
    print_demo_banner()
    
    demos = [
        ("Data Generation", demo_data_generation),
        ("Feature Extraction", demo_feature_extraction),
        ("Model Prediction", demo_model_prediction),
        ("Model Explainability", demo_explainability)
    ]
    
    print("🚀 Starting Fake Account Detection System Demo")
    print("=" * 60)
    
    for demo_name, demo_func in demos:
        print(f"\n🎯 {demo_name}")
        print("=" * 60)
        
        try:
            success = demo_func()
            if success:
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"❌ {demo_name} demo failed")
        except Exception as e:
            print(f"❌ {demo_name} demo error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    
    print("\n📋 What we demonstrated:")
    print("  • Automated generation of realistic fake and real account data")
    print("  • Feature extraction from user profiles and behavior")
    print("  • Machine learning model training and prediction")
    print("  • Explainable AI with SHAP values")
    print("  • Detection of various fake account patterns")
    
    print("\n🚀 To run the complete system:")
    print("  python run_all.py")
    
    print("\n🔧 To test system components:")
    print("  python test_system.py")
    
    print("\n📚 For more information:")
    print("  Read the README.md file")

if __name__ == "__main__":
    main() 