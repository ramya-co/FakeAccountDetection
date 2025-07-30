#!/usr/bin/env python3
"""
Test script to verify the Fake Account Detection System components
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_generation():
    """Test data generation functionality"""
    print("🧪 Testing data generation...")
    
    try:
        from fake_detection.data_generator import DataGenerator
        
        generator = DataGenerator()
        
        # Generate small test datasets
        fake_accounts = generator.generate_fake_accounts(10)
        real_accounts = generator.generate_real_accounts(10)
        
        print(f"✅ Generated {len(fake_accounts)} fake accounts")
        print(f"✅ Generated {len(real_accounts)} real accounts")
        
        # Test data structure
        if fake_accounts and real_accounts:
            fake_account = fake_accounts[0]
            required_fields = ['username', 'bio', 'follower_count', 'following_count', 'post_count', 'is_fake']
            
            for field in required_fields:
                if field not in fake_account:
                    print(f"❌ Missing field: {field}")
                    return False
            
            print("✅ Data structure is correct")
            return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("🧪 Testing feature extraction...")
    
    try:
        from fake_detection.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Test user data
        user_data = {
            'username': 'testuser123',
            'bio': 'Hello world! This is a test bio.',
            'created_at': datetime.now().isoformat(),
            'follower_count': 100,
            'following_count': 50,
            'post_count': 25
        }
        
        features = extractor.extract_features(user_data)
        
        if features:
            print(f"✅ Extracted {len(features)} features")
            print(f"✅ Sample features: {list(features.keys())[:5]}")
            return True
        else:
            print("❌ No features extracted")
            return False
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("🧪 Testing model training...")
    
    try:
        from fake_detection.detector import FakeAccountDetector
        from fake_detection.data_generator import DataGenerator
        
        # Generate small training dataset
        generator = DataGenerator()
        fake_accounts = generator.generate_fake_accounts(50)
        real_accounts = generator.generate_real_accounts(50)
        
        # Create training data
        all_accounts = fake_accounts + real_accounts
        
        # Save to temporary CSV
        df = pd.DataFrame(all_accounts)
        df.to_csv('data/test_training_data.csv', index=False)
        
        # Train model
        detector = FakeAccountDetector()
        X, y = detector.prepare_training_data('data/test_training_data.csv')
        
        if X.shape[0] > 0 and y.shape[0] > 0:
            print(f"✅ Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Test prediction
            test_user = {
                'username': 'testuser',
                'bio': 'Test bio',
                'created_at': datetime.now().isoformat(),
                'follower_count': 10,
                'following_count': 100,
                'post_count': 5
            }
            
            # Train a small model for testing
            detector.train_model(X, y)
            
            # Test prediction
            result = detector.predict_single_user(test_user)
            
            if result and 'is_fake' in result:
                print(f"✅ Model prediction successful: {result['is_fake']}")
                return True
            else:
                print("❌ Model prediction failed")
                return False
        else:
            print("❌ No training data prepared")
            return False
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def test_dashboard_imports():
    """Test dashboard imports"""
    print("🧪 Testing dashboard imports...")
    
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        import requests
        
        print("✅ Dashboard dependencies imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard import test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("🧪 Testing file structure...")
    
    required_files = [
        'requirements.txt',
        'run_all.py',
        'README.md',
        'instagram_clone/app.py',
        'twitter_clone/app.py',
        'fake_detection/detector.py',
        'fake_detection/feature_extractor.py',
        'fake_detection/data_generator.py',
        'dashboard/dashboard.py'
    ]
    
    required_dirs = [
        'instagram_clone/templates',
        'twitter_clone/templates',
        'fake_detection/models',
        'data'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required files and directories exist")
    return True

def main():
    """Run all tests"""
    print("🔍 Fake Account Detection System - Component Tests")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Data Generation", test_data_generation),
        ("Feature Extraction", test_feature_extraction),
        ("Model Training", test_model_training),
        ("Dashboard Imports", test_dashboard_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("\n🚀 To start the system, run:")
        print("   python run_all.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 To fix issues:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check file permissions")
        print("   3. Ensure Python 3.8+ is installed")
    
    # Cleanup test files
    if os.path.exists('data/test_training_data.csv'):
        os.remove('data/test_training_data.csv')

if __name__ == "__main__":
    main() 