#!/usr/bin/env python3
"""
Master script to run the complete Fake Account Detection System
"""

import os
import sys
import time
import subprocess
import threading
import signal
import requests
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print the system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🔍 FAKE ACCOUNT DETECTION SYSTEM 🔍                  ║
    ║                                                              ║
    ║  Instagram Clone | Twitter Clone | ML Detection | Dashboard ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    # Map package names to their import names
    package_imports = {
        'flask': 'flask',
        'flask-sqlalchemy': 'flask_sqlalchemy', 
        'flask-login': 'flask_login',
        'scikit-learn': 'sklearn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'streamlit': 'streamlit',
        'textblob': 'textblob',
        'shap': 'shap',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'requests': 'requests',
        'faker': 'faker'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'data',
        'fake_detection/models',
        'instagram_clone/static',
        'instagram_clone/templates',
        'twitter_clone/static',
        'twitter_clone/templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def train_model():
    """Train the fake account detection model"""
    print("🤖 Training fake account detection model...")
    
    try:
        from fake_detection.detector import FakeAccountDetector
        from fake_detection.data_generator import DataGenerator
        
        # Generate training data if it doesn't exist
        if not os.path.exists('data/training_data.csv'):
            print("📊 Generating training data...")
            generator = DataGenerator()
            generator.generate_training_data(1000, 1000)
        
        # Train model
        detector = FakeAccountDetector()
        X, y = detector.prepare_training_data()
        accuracy = detector.train_model(X, y)
        
        print(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def start_service(service_name, command, port):
    """Start a service in a separate thread"""
    def run_service():
        try:
            print(f"🚀 Starting {service_name} on port {port}...")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for service to start
            time.sleep(3)
            
            # Check if service is running
            try:
                response = requests.get(f"http://localhost:{port}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name} is running on http://localhost:{port}")
                else:
                    print(f"⚠️  {service_name} started but may have issues")
            except requests.exceptions.RequestException:
                print(f"⚠️  {service_name} started but not responding on port {port}")
            
            # Keep the process running
            process.wait()
            
        except Exception as e:
            print(f"❌ Error starting {service_name}: {e}")
    
    thread = threading.Thread(target=run_service, daemon=True)
    thread.start()
    return thread

def wait_for_service(port, service_name, timeout=30):
    """Wait for a service to be ready"""
    print(f"⏳ Waiting for {service_name} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
    
    print(f"⚠️  {service_name} may not be ready yet")
    return False

def main():
    """Main function to run all services"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create directories
    create_directories()
    
    # Train model
    if not train_model():
        print("⚠️  Continuing without trained model...")
    
    print("\n" + "="*60)
    print("🚀 STARTING ALL SERVICES")
    print("="*60)
    
    # Start Instagram clone
    instagram_thread = start_service(
        "Instagram Clone",
        "cd instagram_clone && python app.py",
        5001
    )
    
    # Start Twitter clone
    twitter_thread = start_service(
        "Twitter Clone", 
        "cd twitter_clone && python app.py",
        5002
    )
    
    # Wait for services to start
    time.sleep(5)
    
    # Start dashboard
    dashboard_thread = start_service(
        "Dashboard",
        "streamlit run dashboard/dashboard.py --server.port 8501 --server.headless true",
        8501
    )
    
    # Wait for all services
    time.sleep(3)
    
    print("\n" + "="*60)
    print("🎉 ALL SERVICES STARTED SUCCESSFULLY!")
    print("="*60)
    print()
    print("📱 Instagram Clone:  http://localhost:5001")
    print("   - Username: admin")
    print("   - Password: admin123")
    print()
    print("🐦 Twitter Clone:    http://localhost:5002")
    print("   - Username: admin")
    print("   - Password: admin123")
    print()
    print("📊 Dashboard:        http://localhost:8501")
    print()
    print("🔍 Features:")
    print("   • Create accounts on both platforms")
    print("   • Post content and interact with users")
    print("   • Analyze accounts for fake detection")
    print("   • View detailed analytics in dashboard")
    print("   • Export results and data")
    print()
    print("⏹️  Press Ctrl+C to stop all services")
    print("="*60)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        print("✅ Services stopped. Goodbye!")

def stop_services():
    """Stop all running services"""
    print("🛑 Stopping services...")
    
    # Kill processes on specific ports
    ports = [5001, 5002, 8501]
    
    for port in ports:
        try:
            # Find and kill process using the port
            result = subprocess.run(
                f"lsof -ti:{port} | xargs kill -9",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ Stopped service on port {port}")
        except Exception as e:
            print(f"⚠️  Could not stop service on port {port}: {e}")

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: stop_services())
    
    try:
        main()
    except KeyboardInterrupt:
        stop_services()
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        stop_services()
        sys.exit(1) 