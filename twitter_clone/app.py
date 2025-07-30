from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import os
import random
from faker import Faker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'twitter_clone_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

fake = Faker()

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    bio = db.Column(db.Text, default='')
    profile_pic = db.Column(db.String(200), default='https://via.placeholder.com/150')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_fake = db.Column(db.Boolean, default=False)
    fake_score = db.Column(db.Float, default=0.0)
    
    tweets = db.relationship('Tweet', backref='author', lazy=True)
    followers = db.relationship('Follow', foreign_keys='Follow.followed_id', backref='followed', lazy=True)
    following = db.relationship('Follow', foreign_keys='Follow.follower_id', backref='follower', lazy=True)

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    likes = db.relationship('Like', backref='tweet', lazy=True)
    retweets = db.relationship('Retweet', backref='tweet', lazy=True)

class Follow(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    followed_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tweet_id = db.Column(db.Integer, db.ForeignKey('tweet.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Retweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tweet_id = db.Column(db.Integer, db.ForeignKey('tweet.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            bio=fake.text(max_nb_chars=160),
            profile_pic=f'https://picsum.photos/150/150?random={random.randint(1, 1000)}'
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    # Get tweets from followed users and own tweets
    followed_users = [f.followed_id for f in current_user.following]
    tweets = Tweet.query.filter(Tweet.user_id.in_(followed_users + [current_user.id])).order_by(Tweet.created_at.desc()).all()
    return render_template('home.html', tweets=tweets)

@app.route('/profile/<username>')
@login_required
def profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    tweets = Tweet.query.filter_by(user_id=user.id).order_by(Tweet.created_at.desc()).all()
    is_following = Follow.query.filter_by(follower_id=current_user.id, followed_id=user.id).first() is not None
    return render_template('profile.html', user=user, tweets=tweets, is_following=is_following)

@app.route('/follow/<int:user_id>', methods=['POST'])
@login_required
def follow(user_id):
    if user_id == current_user.id:
        return jsonify({'error': 'Cannot follow yourself'})
    
    existing_follow = Follow.query.filter_by(follower_id=current_user.id, followed_id=user_id).first()
    if existing_follow:
        db.session.delete(existing_follow)
        db.session.commit()
        return jsonify({'status': 'unfollowed'})
    else:
        follow = Follow(follower_id=current_user.id, followed_id=user_id)
        db.session.add(follow)
        db.session.commit()
        return jsonify({'status': 'followed'})

@app.route('/create_tweet', methods=['GET', 'POST'])
@login_required
def create_tweet():
    if request.method == 'POST':
        content = request.form['content']
        if len(content) > 280:
            flash('Tweet must be 280 characters or less')
            return redirect(url_for('create_tweet'))
        
        tweet = Tweet(content=content, user_id=current_user.id)
        db.session.add(tweet)
        db.session.commit()
        
        flash('Tweet posted successfully!')
        return redirect(url_for('home'))
    
    return render_template('create_tweet.html')

@app.route('/like/<int:tweet_id>', methods=['POST'])
@login_required
def like_tweet(tweet_id):
    existing_like = Like.query.filter_by(user_id=current_user.id, tweet_id=tweet_id).first()
    if existing_like:
        db.session.delete(existing_like)
        db.session.commit()
        return jsonify({'status': 'unliked'})
    else:
        like = Like(user_id=current_user.id, tweet_id=tweet_id)
        db.session.add(like)
        db.session.commit()
        return jsonify({'status': 'liked'})

@app.route('/retweet/<int:tweet_id>', methods=['POST'])
@login_required
def retweet(tweet_id):
    existing_retweet = Retweet.query.filter_by(user_id=current_user.id, tweet_id=tweet_id).first()
    if existing_retweet:
        db.session.delete(existing_retweet)
        db.session.commit()
        return jsonify({'status': 'unretweeted'})
    else:
        retweet = Retweet(user_id=current_user.id, tweet_id=tweet_id)
        db.session.add(retweet)
        db.session.commit()
        return jsonify({'status': 'retweeted'})

@app.route('/search')
@login_required
def search():
    query = request.args.get('q', '')
    if query:
        users = User.query.filter(User.username.contains(query)).all()
    else:
        users = []
    return render_template('search.html', users=users, query=query)

@app.route('/admin')
@login_required
def admin():
    if current_user.username != 'admin':
        flash('Access denied')
        return redirect(url_for('home'))
    
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/api/users')
def api_users():
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'bio': user.bio,
        'created_at': user.created_at.isoformat(),
        'is_fake': user.is_fake,
        'fake_score': user.fake_score,
        'follower_count': len(user.followers),
        'following_count': len(user.following),
        'tweet_count': len(user.tweets)
    } for user in users])

@app.route('/api/update_fake_score/<int:user_id>', methods=['POST'])
def update_fake_score(user_id):
    data = request.get_json()
    user = User.query.get_or_404(user_id)
    user.is_fake = data.get('is_fake', False)
    user.fake_score = data.get('fake_score', 0.0)
    db.session.commit()
    return jsonify({'status': 'success'})

def generate_fake_data():
    """Generate fake users and tweets for testing"""
    # Create some fake users
    fake_users = []
    for i in range(50):
        is_fake = random.choice([True, False])
        username = fake.user_name() + str(random.randint(100, 999)) if is_fake else fake.user_name()
        
        user = User(
            username=username,
            email=fake.email(),
            password_hash=generate_password_hash('password123'),
            bio=fake.text(max_nb_chars=160),
            profile_pic=f'https://picsum.photos/150/150?random={random.randint(1, 1000)}',
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365)),
            is_fake=is_fake,
            fake_score=random.uniform(0.0, 1.0) if is_fake else random.uniform(0.0, 0.3)
        )
        db.session.add(user)
        fake_users.append(user)
    
    db.session.commit()
    
    # Create tweets for users
    tweet_templates = [
        "Just had the best coffee ever! ☕",
        "Working on some exciting new projects today! 💻",
        "Beautiful sunset tonight! 🌅",
        "Can't believe how fast time flies!",
        "Great meeting with the team today! 👥",
        "Love this new restaurant I discovered! 🍕",
        "Weekend plans anyone? 🎉",
        "Reading an amazing book right now! 📚",
        "Perfect weather for a walk! 🚶‍♂️",
        "Missing traveling so much! ✈️"
    ]
    
    for user in fake_users:
        for _ in range(random.randint(1, 15)):
            content = random.choice(tweet_templates)
            if len(content) > 280:
                content = content[:277] + "..."
            
            tweet = Tweet(
                content=content,
                user_id=user.id,
                created_at=datetime.utcnow() - timedelta(days=random.randint(0, 30))
            )
            db.session.add(tweet)
    
    db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@twitter.com',
                password_hash=generate_password_hash('admin123'),
                bio='Admin account'
            )
            db.session.add(admin_user)
            db.session.commit()
        
        # Generate fake data if database is empty
        if User.query.count() <= 1:
            generate_fake_data()
    
    app.run(debug=True, port=5002) 