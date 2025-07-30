import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import sys
import os

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
    ["📊 Overview", "👥 User Analysis", "🔍 Single User Analysis", "📁 Batch Analysis", "📈 Model Performance", "⚙️ Settings"]
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

def get_risk_level(score):
    """Get risk level based on fake score"""
    if score >= 0.7:
        return "🔴 HIGH", "red"
    elif score >= 0.4:
        return "🟡 MEDIUM", "orange"
    else:
        return "🟢 LOW", "green"

def get_status_emoji(is_fake):
    """Get status emoji"""
    return "🚨 FAKE" if is_fake else "✅ REAL"

def display_user_analysis_table(users_data, platform_name):
    """Display comprehensive user analysis table"""
    if not users_data:
        st.warning(f"No {platform_name} data available")
        return
    
    df = pd.DataFrame(users_data)
    
    st.subheader(f"👥 {platform_name} User Analysis")
    
    # Create the table with detailed information
    for i, user in enumerate(users_data):
        with st.expander(f"👤 {user['username']} - {get_status_emoji(user['is_fake'])}", expanded=False):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**Username:** {user['username']}")
                st.write(f"**Bio:** {user['bio'][:100]}{'...' if len(user['bio']) > 100 else ''}")
                st.write(f"**Email:** {user['email']}")
            
            with col2:
                score = user['fake_score']
                risk_level, color = get_risk_level(score)
                st.metric("Fake Score", f"{score:.3f}")
                st.write(f"**Risk:** {risk_level}")
            
            with col3:
                st.metric("Followers", user['follower_count'])
                st.metric("Following", user['following_count'])
                st.metric("Posts", user.get('post_count', user.get('tweet_count', 0)))
            
            with col4:
                st.write(f"**Status:** {get_status_emoji(user['is_fake'])}")
                st.write(f"**Created:** {user['created_at'][:10]}")
                
                # Show detailed analysis if available
                if user.get('analysis_data') and user['analysis_data'] != '{}':
                    try:
                        analysis = json.loads(user['analysis_data'])
                        if 'feature_importance' in analysis:
                            st.write("**Top Risk Factors:**")
                            for feature, importance in list(analysis['feature_importance'].items())[:3]:
                                st.write(f"• {feature}: {importance:.3f}")
                    except:
                        pass

def display_detailed_user_card(user_data, platform):
    """Display detailed analysis for a single user"""
    st.subheader(f"🔍 Detailed Analysis: {user_data['username']}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User basic info
        st.write(f"**Platform:** {platform}")
        st.write(f"**Username:** {user_data['username']}")
        st.write(f"**Bio:** {user_data['bio']}")
        st.write(f"**Email:** {user_data['email']}")
        st.write(f"**Created:** {user_data['created_at']}")
        
        # Network info
        st.write(f"**Followers:** {user_data['follower_count']}")
        st.write(f"**Following:** {user_data['following_count']}")
        st.write(f"**Posts:** {user_data.get('post_count', user_data.get('tweet_count', 0))}")
    
    with col2:
        # Fake detection results
        score = user_data['fake_score']
        risk_level, color = get_risk_level(score)
        
        st.metric("Fake Score", f"{score:.3f}", delta=None)
        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Status:** {get_status_emoji(user_data['is_fake'])}")
        
        # Progress bar for fake score
        st.progress(score)
    
    # Detailed analysis breakdown
    if user_data.get('analysis_data') and user_data['analysis_data'] != '{}':
        try:
            analysis = json.loads(user_data['analysis_data'])
            
            st.markdown("---")
            st.subheader("📊 Detailed Analysis Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔤 Username Analysis:**")
                username_features = {k: v for k, v in analysis.get('feature_importance', {}).items() 
                                   if 'username' in k.lower()}
                for feature, importance in sorted(username_features.items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {feature}: {importance:.3f}")
                
                st.write("**💬 Bio Analysis:**")
                bio_features = {k: v for k, v in analysis.get('feature_importance', {}).items() 
                              if 'bio' in k.lower() or 'sentiment' in k.lower()}
                for feature, importance in sorted(bio_features.items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {feature}: {importance:.3f}")
            
            with col2:
                st.write("**👥 Network Analysis:**")
                network_features = {k: v for k, v in analysis.get('feature_importance', {}).items() 
                                  if 'follower' in k.lower() or 'following' in k.lower() or 'network' in k.lower()}
                for feature, importance in sorted(network_features.items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {feature}: {importance:.3f}")
                
                st.write("**📅 Account Analysis:**")
                account_features = {k: v for k, v in analysis.get('feature_importance', {}).items() 
                                  if 'age' in k.lower() or 'created' in k.lower() or 'activity' in k.lower()}
                for feature, importance in sorted(account_features.items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {feature}: {importance:.3f}")
            
            # Feature importance chart
            if 'feature_importance' in analysis:
                st.markdown("---")
                st.subheader("📈 Feature Importance")
                
                features = list(analysis['feature_importance'].keys())[:10]
                importance = list(analysis['feature_importance'].values())[:10]
                
                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    title="Top Contributing Features",
                    labels={'x': 'Importance', 'y': 'Feature'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error parsing analysis data: {e}")

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
                labels={'follower_count': 'Followers', 'following_count': 'Following'}
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.scatter(
                combined_df,
                x='fake_score',
                y='follower_count',
                color='platform',
                title='Fake Score vs Followers',
                labels={'fake_score': 'Fake Score', 'follower_count': 'Followers'}
            )
            st.plotly_chart(fig4, use_container_width=True)

# User Analysis Page
elif page == "👥 User Analysis":
    st.title("👥 Real-Time User Analysis")
    st.markdown("---")
    
    # Fetch data
    instagram_data = fetch_data_from_api(INSTAGRAM_API)
    twitter_data = fetch_data_from_api(TWITTER_API)
    
    # Display user analysis tables
    display_user_analysis_table(instagram_data, "Instagram")
    st.markdown("---")
    display_user_analysis_table(twitter_data, "Twitter")

# Single User Analysis Page
elif page == "🔍 Single User Analysis":
    st.title("🔍 Single User Analysis")
    st.markdown("---")
    
    # Manual user input
    st.subheader("Enter User Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input("Username", placeholder="e.g., user123456")
        bio = st.text_area("Bio", placeholder="Enter user bio...")
    
    with col2:
        follower_count = st.number_input("Followers", min_value=0, value=0)
        following_count = st.number_input("Following", min_value=0, value=0)
        post_count = st.number_input("Posts", min_value=0, value=0)
    
    if st.button("🔍 Analyze User"):
        if username:
            user_data = {
                'username': username,
                'bio': bio,
                'follower_count': follower_count,
                'following_count': following_count,
                'post_count': post_count,
                'created_at': datetime.now().isoformat()
            }
            
            result = analyze_user(user_data)
            if result:
                st.success("Analysis completed!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fake Score", f"{result['fake_probability']:.3f}")
                
                with col2:
                    status = "🚨 FAKE" if result['is_fake'] else "✅ REAL"
                    st.metric("Status", status)
                
                with col3:
                    risk_level, _ = get_risk_level(result['fake_probability'])
                    st.metric("Risk Level", risk_level)
                
                # Progress bar
                st.progress(result['fake_probability'])
                
                # Feature importance
                if 'explanation' in result and 'feature_importance' in result['explanation']:
                    st.subheader("📊 Top Contributing Features")
                    
                    features = list(result['explanation']['feature_importance'].keys())[:10]
                    importance = list(result['explanation']['feature_importance'].values())[:10]
                    
                    fig = px.bar(
                        x=importance,
                        y=features,
                        orientation='h',
                        title="Feature Importance",
                        labels={'x': 'Importance', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Batch Analysis Page
elif page == "📁 Batch Analysis":
    st.title("📁 Batch Analysis")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV file with user data", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.dataframe(df.head())
        
        if st.button("🔍 Analyze Batch"):
            results = []
            
            for _, row in df.iterrows():
                user_data = {
                    'username': row.get('username', ''),
                    'bio': row.get('bio', ''),
                    'follower_count': row.get('follower_count', 0),
                    'following_count': row.get('following_count', 0),
                    'post_count': row.get('post_count', 0),
                    'created_at': row.get('created_at', datetime.now().isoformat())
                }
                
                result = analyze_user(user_data)
                if result:
                    results.append({
                        'username': user_data['username'],
                        'fake_score': result['fake_probability'],
                        'is_fake': result['is_fake'],
                        'status': get_status_emoji(result['is_fake'])
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                st.success(f"Analysis completed for {len(results)} users!")
                
                # Display results
                st.subheader("📊 Batch Analysis Results")
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="batch_analysis_results.csv",
                    mime="text/csv"
                )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        results_df,
                        x='fake_score',
                        title='Fake Score Distribution',
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fake_count = len(results_df[results_df['is_fake'] == True])
                    real_count = len(results_df[results_df['is_fake'] == False])
                    
                    fig = px.pie(
                        values=[fake_count, real_count],
                        names=['Fake', 'Real'],
                        title='Fake vs Real Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Model Performance Page
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")
    
    if detector:
        # Feature importance
        feature_importance = detector.get_feature_importance()
        if feature_importance:
            st.subheader("🔍 Feature Importance")
            
            # Get top 15 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            features, importance = zip(*top_features)
            
            fig = px.bar(
                x=importance,
                y=features,
                orientation='h',
                title="Top 15 Most Important Features",
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model evaluation
        if st.button("📊 Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    accuracy, report = detector.evaluate_model()
                    st.success(f"Model Accuracy: {accuracy:.2%}")
                    
                    st.subheader("📋 Classification Report")
                    st.text(report)
                    
                except Exception as e:
                    st.error(f"Error evaluating model: {e}")

# Settings Page
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 System Actions")
        
        if st.button("🤖 Retrain Model"):
            with st.spinner("Retraining model..."):
                try:
                    X, y = detector.prepare_training_data()
                    accuracy = detector.train_model(X, y)
                    st.success(f"Model retrained! Accuracy: {accuracy:.2%}")
                except Exception as e:
                    st.error(f"Error retraining model: {e}")
        
        if st.button("📊 Generate Test Data"):
            with st.spinner("Generating test data..."):
                try:
                    from fake_detection.data_generator import DataGenerator
                    generator = DataGenerator()
                    generator.generate_training_data(1000, 1000)
                    st.success("Test data generated successfully!")
                except Exception as e:
                    st.error(f"Error generating test data: {e}")
    
    with col2:
        st.subheader("📁 Data Files")
        
        data_files = [
            "data/fake_accounts.json",
            "data/real_accounts.json", 
            "data/training_data.csv",
            "fake_detection/models/fake_detector_model.pkl"
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                st.write(f"✅ {file_path}")
            else:
                st.write(f"❌ {file_path}")
        
        st.subheader("🔗 API Endpoints")
        st.write(f"Instagram: {INSTAGRAM_API}")
        st.write(f"Twitter: {TWITTER_API}") 