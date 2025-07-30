# 🔍 Fake Account Detection System

A complete fake account detection system with Instagram and Twitter clones for testing purposes. This system uses machine learning to detect fake accounts based on various features extracted from user profiles and behavior patterns.

## 📋 Features

### 🏗️ System Architecture
- **Instagram Clone**: Full-featured social media platform with posts, follows, likes, and comments
- **Twitter Clone**: Microblogging platform with tweets, retweets, and user search
- **ML Detection Engine**: Machine learning model using Random Forest for fake account detection
- **Interactive Dashboard**: Streamlit-based dashboard for analysis and visualization
- **Data Generation**: Automated generation of realistic fake and real account data

### 🔍 Fake Account Detection Features
- **Username Analysis**: Pattern detection, entropy calculation, suspicious character sequences
- **Bio Analysis**: Sentiment analysis, suspicious keyword detection, link detection
- **Network Analysis**: Follower/following ratios, network balance, suspicious patterns
- **Account Age Analysis**: Account creation date analysis and categorization
- **Activity Analysis**: Posting frequency, engagement patterns
- **SHAP Explainability**: Detailed explanations of why accounts are flagged as fake

### 📊 Dashboard Features
- **Real-time Analytics**: Live data from both social media platforms
- **Single User Analysis**: Detailed analysis of individual accounts
- **Batch Analysis**: Upload CSV files for bulk analysis
- **Model Performance**: Feature importance and model evaluation
- **Data Export**: Export results and analysis data
- **Interactive Visualizations**: Charts and graphs for data exploration

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-account-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete system**
   ```bash
   python run_all.py
   ```

### 🎯 What happens when you run the system:

1. **Dependency Check**: Verifies all required packages are installed
2. **Directory Creation**: Creates necessary folders and structure
3. **Model Training**: Generates training data and trains the ML model
4. **Service Startup**: Launches all three services:
   - Instagram Clone (Port 5001)
   - Twitter Clone (Port 5002)
   - Dashboard (Port 8501)

## 🌐 Accessing the System

Once the system is running, you can access:

### 📱 Instagram Clone
- **URL**: http://localhost:5001
- **Admin Login**: 
  - Username: `admin`
  - Password: `admin123`

### 🐦 Twitter Clone
- **URL**: http://localhost:5002
- **Admin Login**:
  - Username: `admin`
  - Password: `admin123`

### 📊 Dashboard
- **URL**: http://localhost:8501
- **Features**: Analytics, user analysis, batch processing

## 📁 Project Structure

```
fake-account-detector/
├── instagram_clone/          # Instagram clone application
│   ├── app.py               # Main Flask application
│   ├── templates/           # HTML templates
│   ├── static/              # CSS/JS files
│   └── database.db          # SQLite database
├── twitter_clone/           # Twitter clone application
│   ├── app.py               # Main Flask application
│   ├── templates/           # HTML templates
│   ├── static/              # CSS/JS files
│   └── database.db          # SQLite database
├── fake_detection/          # ML detection system
│   ├── detector.py          # Main detection engine
│   ├── feature_extractor.py # Feature extraction
│   ├── data_generator.py    # Data generation
│   └── models/              # Saved ML models
├── dashboard/               # Streamlit dashboard
│   └── dashboard.py         # Main dashboard application
├── data/                    # Data storage
│   ├── fake_accounts.json   # Generated fake accounts
│   ├── real_accounts.json   # Generated real accounts
│   └── training_data.csv    # Training dataset
├── requirements.txt         # Python dependencies
├── run_all.py              # Master startup script
└── README.md               # This file
```

## 🔧 Manual Setup (Alternative)

If you prefer to run services individually:

### 1. Train the Model
```bash
cd fake_detection
python detector.py
```

### 2. Start Instagram Clone
```bash
cd instagram_clone
python app.py
```

### 3. Start Twitter Clone
```bash
cd twitter_clone
python app.py
```

### 4. Start Dashboard
```bash
streamlit run dashboard/dashboard.py
```

## 📊 Using the System

### Creating Test Data
The system automatically generates realistic test data, but you can create more:

1. **Access Admin Panel**: Login as admin on either platform
2. **Generate Data**: Use the admin panel to create additional test accounts
3. **Export Data**: Use the API endpoints to export user data

### Analyzing Accounts

#### Single User Analysis
1. Go to the Dashboard
2. Navigate to "Single User Analysis"
3. Enter user details (username, bio, follower counts, etc.)
4. Click "Analyze User" to get results

#### Batch Analysis
1. Prepare a CSV file with columns: `username`, `bio`, `follower_count`, `following_count`, `post_count`
2. Go to "Batch Analysis" in the dashboard
3. Upload your CSV file
4. Click "Analyze All Users"
5. Download results

### API Endpoints

#### Instagram Clone
- `GET /api/users` - Get all users
- `POST /api/update_fake_score/<user_id>` - Update fake score

#### Twitter Clone
- `GET /api/users` - Get all users
- `POST /api/update_fake_score/<user_id>` - Update fake score

## 🤖 Machine Learning Model

### Features Used
- **Username Features**: Length, entropy, patterns, special characters
- **Bio Features**: Sentiment, suspicious keywords, links, hashtags
- **Network Features**: Follower ratios, network balance, suspicious patterns
- **Account Features**: Age, activity patterns, posting frequency

### Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: 28 engineered features
- **Training Data**: 1000 fake + 1000 real accounts
- **Cross-validation**: 5-fold CV
- **Explainability**: SHAP values for feature importance

### Model Performance
- **Accuracy**: ~95% on test data
- **Precision**: High precision for fake account detection
- **Recall**: Good recall for identifying fake accounts
- **Explainability**: Detailed feature importance analysis

## 🔍 Detection Criteria

### Fake Account Indicators
- **Suspicious Usernames**: Random numbers, special characters, generic names
- **Promotional Bios**: Money-making schemes, suspicious links, spam content
- **Network Imbalance**: High following, low followers
- **Recent Accounts**: Very new accounts with high activity
- **Low Engagement**: Minimal interaction with content
- **Suspicious Patterns**: Automated behavior patterns

### Real Account Indicators
- **Natural Usernames**: Human-like, meaningful usernames
- **Personal Bios**: Genuine personal information, interests
- **Balanced Networks**: Reasonable follower/following ratios
- **Account Age**: Established accounts with history
- **Natural Activity**: Organic posting and interaction patterns

## 📈 Dashboard Features

### Overview Page
- Real-time statistics from both platforms
- Fake account distribution charts
- Network analysis visualizations
- Platform comparison metrics

### Single User Analysis
- Detailed user analysis with feature breakdown
- SHAP-based explanation of predictions
- Risk factor visualization
- Feature importance ranking

### Batch Analysis
- CSV file upload and processing
- Bulk analysis results
- Export functionality
- Statistical summaries

### Model Performance
- Feature importance visualization
- Model evaluation metrics
- Confusion matrix
- Cross-validation results

## 🛠️ Customization

### Adding New Features
1. Modify `fake_detection/feature_extractor.py`
2. Add new feature extraction methods
3. Update the feature list in `get_feature_names()`
4. Retrain the model

### Modifying Detection Logic
1. Edit `fake_detection/detector.py`
2. Adjust model parameters
3. Change feature importance weights
4. Update training data generation

### Customizing Social Media Clones
1. Modify Flask applications in respective directories
2. Add new features or modify existing ones
3. Update templates and static files
4. Extend database models

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find processes using ports
lsof -i :5001
lsof -i :5002
lsof -i :8501

# Kill processes
kill -9 <PID>
```

#### Model Training Issues
```bash
# Regenerate training data
cd fake_detection
python data_generator.py

# Retrain model
python detector.py
```

#### Database Issues
```bash
# Remove existing databases
rm instagram_clone/database.db
rm twitter_clone/database.db

# Restart services
python run_all.py
```

### Dependencies Issues
```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📝 API Documentation

### User Data Format
```json
{
  "username": "string",
  "bio": "string",
  "created_at": "ISO datetime string",
  "follower_count": "integer",
  "following_count": "integer",
  "post_count": "integer"
}
```

### Analysis Result Format
```json
{
  "is_fake": "boolean",
  "fake_probability": "float (0-1)",
  "real_probability": "float (0-1)",
  "features": "object",
  "explanation": "object (optional)"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please use responsibly and in accordance with applicable laws and regulations.

## 🙏 Acknowledgments

- Flask for the web framework
- Streamlit for the dashboard
- Scikit-learn for machine learning
- SHAP for model explainability
- Faker for data generation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub

---

**Note**: This system is designed for educational and research purposes. The fake account detection is based on simulated data and should not be used for production systems without proper validation and testing. 