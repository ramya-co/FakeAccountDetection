import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import sys
from datetime import datetime
import time

# Add parent directory to path to import fake detection modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fake_detection.detector import FakeAccountDetector
from fake_detection.feature_extractor import FeatureExtractor

# Page configuration
st.set_page_config(
    page_title="Fake Account Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize detector
@st.cache_resource
def load_detector():
    detector = FakeAccountDetector()
    if not detector.load_model():
        st.error("No trained model found. Please train the model first.")
        return None
    return detector

detector = load_detector()

# Sidebar
st.sidebar.title("🔍 Fake Account Detection")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["📊 Overview", "🔍 Single User Analysis", "📁 Batch Analysis", "📈 Model Performance", "⚙️ Settings"]
)

# API endpoints
INSTAGRAM_API = "http://localhost:5001/api/users"
TWITTER_API = "http://localhost:5002/api/users"

def fetch_data_from_api(api_url):
    """Fetch data from API endpoint"""
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch data from {api_url}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to {api_url}: {e}")
        return []

def analyze_user(user_data):
    """Analyze a single user"""
    if detector is None:
        return None
    
    try:
        result = detector.predict_single_user(user_data)
        explanation = detector.explain_prediction(user_data)
        result['explanation'] = explanation
        return result
    except Exception as e:
        st.error(f"Error analyzing user: {e}")
        return None

# Overview Page
if page == "📊 Overview":
    st.title("📊 Fake Account Detection Overview")
    st.markdown("---")
    
    # Fetch data from both platforms
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Instagram Data")
        instagram_data = fetch_data_from_api(INSTAGRAM_API)
        if instagram_data:
            instagram_df = pd.DataFrame(instagram_data)
            st.metric("Total Users", len(instagram_df))
            st.metric("Fake Accounts", len(instagram_df[instagram_df['is_fake'] == True]))
            st.metric("Fake Percentage", f"{len(instagram_df[instagram_df['is_fake'] == True]) / len(instagram_df) * 100:.1f}%")
        else:
            st.warning("Instagram data not available")
    
    with col2:
        st.subheader("🐦 Twitter Data")
        twitter_data = fetch_data_from_api(TWITTER_API)
        if twitter_data:
            twitter_df = pd.DataFrame(twitter_data)
            st.metric("Total Users", len(twitter_df))
            st.metric("Fake Accounts", len(twitter_df[twitter_df['is_fake'] == True]))
            st.metric("Fake Percentage", f"{len(twitter_df[twitter_df['is_fake'] == True]) / len(twitter_df) * 100:.1f}%")
        else:
            st.warning("Twitter data not available")
    
    # Combined statistics
    st.markdown("---")
    st.subheader("📈 Combined Statistics")
    
    if instagram_data and twitter_data:
        combined_df = pd.concat([
            pd.DataFrame(instagram_data).assign(platform='Instagram'),
            pd.DataFrame(twitter_data).assign(platform='Twitter')
        ], ignore_index=True)
        
        # Fake account distribution by platform
        fig = px.pie(
            combined_df[combined_df['is_fake'] == True],
            names='platform',
            title='Fake Account Distribution by Platform'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fake score distribution
        fig2 = px.histogram(
            combined_df,
            x='fake_score',
            color='platform',
            title='Fake Score Distribution',
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Network analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.scatter(
                combined_df,
                x='follower_count',
                y='following_count',
                color='is_fake',
                title='Followers vs Following',
                hover_data=['username', 'platform']
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.scatter(
                combined_df,
                x='fake_score',
                y='follower_count',
                color='platform',
                title='Fake Score vs Followers',
                hover_data=['username']
            )
            st.plotly_chart(fig4, use_container_width=True)

# Single User Analysis Page
elif page == "🔍 Single User Analysis":
    st.title("🔍 Single User Analysis")
    st.markdown("---")
    
    # User input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter User Data")
        
        username = st.text_input("Username", placeholder="Enter username")
        bio = st.text_area("Bio", placeholder="Enter user bio")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            follower_count = st.number_input("Followers", min_value=0, value=100)
            post_count = st.number_input("Posts", min_value=0, value=10)
        
        with col1_2:
            following_count = st.number_input("Following", min_value=0, value=50)
            account_age_days = st.number_input("Account Age (days)", min_value=0, value=30)
    
    with col2:
        st.subheader("Analysis Results")
        
        if st.button("🔍 Analyze User", type="primary"):
            if username:
                user_data = {
                    'username': username,
                    'bio': bio,
                    'created_at': (datetime.now() - pd.Timedelta(days=account_age_days)).isoformat(),
                    'follower_count': follower_count,
                    'following_count': following_count,
                    'post_count': post_count
                }
                
                result = analyze_user(user_data)
                
                if result:
                    # Display results
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        if result['is_fake']:
                            st.error("🚨 FAKE ACCOUNT DETECTED")
                        else:
                            st.success("✅ LEGITIMATE ACCOUNT")
                        
                        st.metric("Fake Probability", f"{result['fake_probability']:.2%}")
                        st.metric("Real Probability", f"{result['real_probability']:.2%}")
                    
                    with col2_2:
                        st.metric("Followers", follower_count)
                        st.metric("Following", following_count)
                        st.metric("Posts", post_count)
                    
                    # Feature importance
                    st.subheader("🔍 Feature Analysis")
                    if 'explanation' in result:
                        importance = result['explanation']['feature_importance']
                        
                        # Create feature importance chart
                        fig = px.bar(
                            x=list(importance.values()),
                            y=list(importance.keys()),
                            orientation='h',
                            title='Top Contributing Features'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed features
                        st.subheader("📊 Detailed Features")
                        features = result['features']
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.write("**Username Features:**")
                            st.write(f"- Length: {features.get('username_length', 0)}")
                            st.write(f"- Entropy: {features.get('username_entropy', 0):.2f}")
                            st.write(f"- Has numbers: {features.get('username_has_numbers', 0)}")
                            st.write(f"- Has special chars: {features.get('username_has_special_chars', 0)}")
                        
                        with col4:
                            st.write("**Network Features:**")
                            st.write(f"- Follower ratio: {features.get('follower_following_ratio', 0):.2f}")
                            st.write(f"- Network balance: {features.get('network_balance', 0):.2f}")
                            st.write(f"- High following/low followers: {features.get('high_following_low_followers', 0)}")
                            st.write(f"- Zero followers: {features.get('zero_followers', 0)}")
            else:
                st.warning("Please enter a username to analyze")

# Batch Analysis Page
elif page == "📁 Batch Analysis":
    st.title("📁 Batch Analysis")
    st.markdown("---")
    
    # File upload
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with user data",
        type=['csv'],
        help="CSV should have columns: username, bio, follower_count, following_count, post_count"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} users from file")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Users", type="primary"):
                if detector is None:
                    st.error("No trained model available")
                else:
                    # Prepare data for analysis
                    users_data = []
                    for _, row in df.iterrows():
                        user_data = {
                            'username': row['username'],
                            'bio': row.get('bio', ''),
                            'created_at': row.get('created_at', datetime.now().isoformat()),
                            'follower_count': row['follower_count'],
                            'following_count': row['following_count'],
                            'post_count': row['post_count']
                        }
                        users_data.append(user_data)
                    
                    # Analyze users
                    with st.spinner("Analyzing users..."):
                        results = detector.predict_batch(users_data)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Users", len(results_df))
                    with col2:
                        fake_count = len(results_df[results_df['is_fake'] == True])
                        st.metric("Fake Accounts", fake_count)
                    with col3:
                        fake_percentage = fake_count / len(results_df) * 100
                        st.metric("Fake Percentage", f"{fake_percentage:.1f}%")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fake_account_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            results_df,
                            x='fake_probability',
                            title='Fake Probability Distribution',
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig2 = px.scatter(
                            results_df,
                            x='follower_count',
                            y='fake_probability',
                            color='is_fake',
                            title='Followers vs Fake Probability',
                            hover_data=['username']
                        )
                        st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Model Performance Page
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")
    
    if detector is None:
        st.error("No trained model available")
    else:
        # Feature importance
        st.subheader("🔍 Feature Importance")
        importance = detector.get_feature_importance()
        
        fig = px.bar(
            x=list(importance.values())[:15],
            y=list(importance.keys())[:15],
            orientation='h',
            title='Top 15 Most Important Features'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model evaluation
        st.subheader("📊 Model Evaluation")
        if st.button("🔄 Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    evaluation = detector.evaluate_model()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{evaluation['accuracy']:.2%}")
                    
                    with col2:
                        st.metric("Test Samples", len(evaluation['true_labels']))
                    
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(evaluation['true_labels'], evaluation['predictions'])
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Real', 'Fake'],
                        y=['Real', 'Fake']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error evaluating model: {e}")

# Settings Page
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    st.subheader("🔧 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retrain Model", type="primary"):
            if detector is None:
                st.error("Detector not initialized")
            else:
                with st.spinner("Training model..."):
                    try:
                        X, y = detector.prepare_training_data()
                        accuracy = detector.train_model(X, y)
                        st.success(f"✅ Model retrained successfully! Accuracy: {accuracy:.2%}")
                    except Exception as e:
                        st.error(f"Error training model: {e}")
    
    with col2:
        if st.button("📊 Generate Test Data"):
            try:
                from fake_detection.data_generator import DataGenerator
                generator = DataGenerator()
                test_data = generator.generate_test_data(100)
                
                df = pd.DataFrame(test_data)
                df.to_csv('data/test_data.csv', index=False)
                st.success("✅ Test data generated successfully!")
            except Exception as e:
                st.error(f"Error generating test data: {e}")
    
    st.subheader("📁 Data Management")
    
    # Show data files
    data_files = []
    if os.path.exists('data'):
        for file in os.listdir('data'):
            if file.endswith(('.csv', '.json')):
                file_path = os.path.join('data', file)
                size = os.path.getsize(file_path)
                data_files.append({
                    'File': file,
                    'Size': f"{size / 1024:.1f} KB",
                    'Modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
                })
    
    if data_files:
        st.dataframe(pd.DataFrame(data_files))
    else:
        st.info("No data files found in data/ directory")
    
    st.subheader("🔗 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Instagram API:**")
        st.code("http://localhost:5001/api/users")
    
    with col2:
        st.write("**Twitter API:**")
        st.code("http://localhost:5002/api/users")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Fake Account Detection System | Built with Streamlit and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
) 