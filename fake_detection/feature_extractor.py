import re
import math
from datetime import datetime
from textblob import TextBlob
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            'buy', 'sell', 'earn', 'money', 'profit', 'investment', 'crypto', 'bitcoin',
            'click', 'link', 'offer', 'discount', 'free', 'limited', 'urgent', 'act now',
            'make money', 'work from home', 'get rich', 'quick cash', 'easy money'
        ]
        
        self.suspicious_patterns = [
            r'\d{4,}',  # 4+ consecutive digits
            r'[a-zA-Z]{10,}',  # 10+ consecutive letters
            r'[!@#$%^&*]{3,}',  # 3+ consecutive special characters
            r'(.)\1{3,}',  # 4+ consecutive same characters
        ]
    
    def extract_features(self, user_data):
        """Extract features from user data for fake account detection"""
        features = {}
        
        # Username features
        features.update(self._extract_username_features(user_data.get('username', '')))
        
        # Bio features
        features.update(self._extract_bio_features(user_data.get('bio', '')))
        
        # Account age features
        features.update(self._extract_account_age_features(user_data.get('created_at')))
        
        # Network features
        features.update(self._extract_network_features(
            user_data.get('follower_count', 0),
            user_data.get('following_count', 0),
            user_data.get('post_count', 0)
        ))
        
        # Activity features
        features.update(self._extract_activity_features(user_data))
        
        return features
    
    def _extract_username_features(self, username):
        """Extract features from username"""
        features = {}
        
        if not username:
            return features
        
        # Length features
        features['username_length'] = len(username)
        features['username_entropy'] = self._calculate_entropy(username)
        
        # Pattern features
        features['username_has_numbers'] = int(bool(re.search(r'\d', username)))
        features['username_has_special_chars'] = int(bool(re.search(r'[!@#$%^&*]', username)))
        features['username_has_consecutive_numbers'] = int(bool(re.search(r'\d{3,}', username)))
        features['username_has_consecutive_chars'] = int(bool(re.search(r'(.)\1{2,}', username)))
        
        # Suspicious patterns
        features['username_suspicious_patterns'] = sum(
            int(bool(re.search(pattern, username))) 
            for pattern in self.suspicious_patterns
        )
        
        # Dictionary word check
        features['username_is_dictionary_word'] = int(self._is_dictionary_word(username.lower()))
        
        return features
    
    def _extract_bio_features(self, bio):
        """Extract features from bio text"""
        features = {}
        
        if not bio:
            features.update({
                'bio_length': 0,
                'bio_sentiment': 0,
                'bio_has_links': 0,
                'bio_has_suspicious_keywords': 0,
                'bio_word_count': 0,
                'bio_hashtag_count': 0,
                'bio_mention_count': 0
            })
            return features
        
        # Length and word count
        features['bio_length'] = len(bio)
        features['bio_word_count'] = len(bio.split())
        
        # Sentiment analysis
        blob = TextBlob(bio)
        features['bio_sentiment'] = blob.sentiment.polarity
        
        # Link detection
        features['bio_has_links'] = int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', bio)))
        
        # Suspicious keywords
        bio_lower = bio.lower()
        features['bio_has_suspicious_keywords'] = sum(
            int(keyword in bio_lower) 
            for keyword in self.suspicious_keywords
        )
        
        # Hashtag and mention counts
        features['bio_hashtag_count'] = len(re.findall(r'#\w+', bio))
        features['bio_mention_count'] = len(re.findall(r'@\w+', bio))
        
        return features
    
    def _extract_account_age_features(self, created_at):
        """Extract features from account creation date"""
        features = {}
        
        if not created_at:
            features['account_age_days'] = 0
            features['account_age_category'] = 0
            return features
        
        # Parse date
        if isinstance(created_at, str):
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            created_date = created_at
        
        # Calculate age
        age_days = (datetime.now() - created_date).days
        features['account_age_days'] = age_days
        
        # Age categories
        if age_days < 7:
            features['account_age_category'] = 0  # Very new
        elif age_days < 30:
            features['account_age_category'] = 1  # New
        elif age_days < 90:
            features['account_age_category'] = 2  # Recent
        elif age_days < 365:
            features['account_age_category'] = 3  # Established
        else:
            features['account_age_category'] = 4  # Old
        
        return features
    
    def _extract_network_features(self, follower_count, following_count, post_count):
        """Extract features from network statistics"""
        features = {}
        
        # Basic counts
        features['follower_count'] = follower_count
        features['following_count'] = following_count
        features['post_count'] = post_count
        
        # Ratios
        if following_count > 0:
            features['follower_following_ratio'] = follower_count / following_count
        else:
            features['follower_following_ratio'] = follower_count
        
        if follower_count > 0:
            features['following_follower_ratio'] = following_count / follower_count
        else:
            features['following_follower_ratio'] = following_count
        
        # Total network size
        features['total_network_size'] = follower_count + following_count
        
        # Network balance
        if features['total_network_size'] > 0:
            features['network_balance'] = abs(follower_count - following_count) / features['total_network_size']
        else:
            features['network_balance'] = 0
        
        # Suspicious patterns
        features['high_following_low_followers'] = int(following_count > 100 and follower_count < 10)
        features['zero_followers'] = int(follower_count == 0)
        features['zero_following'] = int(following_count == 0)
        features['zero_posts'] = int(post_count == 0)
        
        return features
    
    def _extract_activity_features(self, user_data):
        """Extract activity-related features"""
        features = {}
        
        # Post frequency (if we have post dates)
        posts = user_data.get('posts', [])
        if posts and len(posts) > 1:
            # Calculate average time between posts
            post_dates = sorted([post.get('created_at') for post in posts if post.get('created_at')])
            if len(post_dates) > 1:
                intervals = []
                for i in range(1, len(post_dates)):
                    if isinstance(post_dates[i], str):
                        date1 = datetime.fromisoformat(post_dates[i-1].replace('Z', '+00:00'))
                        date2 = datetime.fromisoformat(post_dates[i].replace('Z', '+00:00'))
                    else:
                        date1 = post_dates[i-1]
                        date2 = post_dates[i]
                    intervals.append((date2 - date1).days)
                
                features['avg_post_interval'] = np.mean(intervals) if intervals else 0
                features['post_interval_variance'] = np.var(intervals) if len(intervals) > 1 else 0
            else:
                features['avg_post_interval'] = 0
                features['post_interval_variance'] = 0
        else:
            features['avg_post_interval'] = 0
            features['post_interval_variance'] = 0
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        length = len(text)
        for count in char_counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _is_dictionary_word(self, word):
        """Check if a word is a common dictionary word"""
        # Simple implementation - in production, use a proper dictionary
        common_words = {
            'user', 'admin', 'test', 'demo', 'guest', 'anonymous', 'unknown',
            'john', 'jane', 'mike', 'sarah', 'david', 'lisa', 'chris', 'emma'
        }
        return word in common_words
    
    def get_feature_names(self):
        """Get list of all feature names"""
        return [
            'username_length', 'username_entropy', 'username_has_numbers',
            'username_has_special_chars', 'username_has_consecutive_numbers',
            'username_has_consecutive_chars', 'username_suspicious_patterns',
            'username_is_dictionary_word', 'bio_length', 'bio_sentiment',
            'bio_has_links', 'bio_has_suspicious_keywords', 'bio_word_count',
            'bio_hashtag_count', 'bio_mention_count', 'account_age_days',
            'account_age_category', 'follower_count', 'following_count',
            'post_count', 'follower_following_ratio', 'following_follower_ratio',
            'total_network_size', 'network_balance', 'high_following_low_followers',
            'zero_followers', 'zero_following', 'zero_posts',
            'avg_post_interval', 'post_interval_variance'
        ] 