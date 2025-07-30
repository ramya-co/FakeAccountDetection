import json
import random
from datetime import datetime, timedelta
from faker import Faker
import pandas as pd
import numpy as np

class DataGenerator:
    def __init__(self):
        self.fake = Faker()
        
    def generate_fake_accounts(self, count=1000):
        """Generate fake account data with suspicious patterns"""
        fake_accounts = []
        
        for i in range(count):
            # Generate suspicious username patterns
            username_patterns = [
                lambda: f"{self.fake.user_name()}{random.randint(1000, 9999)}",
                lambda: f"{self.fake.user_name()}{random.choice(['123', '456', '789', '000'])}",
                lambda: f"{self.fake.user_name()}{random.choice(['_', '.', ''])}{random.randint(10, 99)}",
                lambda: f"{self.fake.user_name()}{random.choice(['bot', 'fake', 'spam', 'test'])}",
                lambda: f"{random.choice(['user', 'admin', 'test', 'demo'])}{random.randint(100, 999)}",
                lambda: f"{self.fake.user_name()}{random.choice(['!', '@', '#', '$'])}{random.randint(1, 9)}"
            ]
            
            username = random.choice(username_patterns)()
            
            # Generate suspicious bio patterns
            bio_patterns = [
                f"💰 Make money fast! Click here: {self.fake.url()}",
                f"🔥 Limited time offer! {self.fake.url()}",
                f"💸 Earn {random.randint(100, 1000)}$ daily! {self.fake.url()}",
                f"🚀 Join our crypto investment program! {self.fake.url()}",
                f"📱 Download our app and get rich! {self.fake.url()}",
                f"🎁 Free giveaway! Click to win! {self.fake.url()}",
                f"💯 100% guaranteed profit! {self.fake.url()}",
                f"⚡ Quick cash method! {self.fake.url()}",
                f"🏠 Work from home! Earn {random.randint(50, 500)}$/hour! {self.fake.url()}",
                f"🎯 Best investment opportunity! {self.fake.url()}"
            ]
            
            bio = random.choice(bio_patterns)
            
            # Generate suspicious network patterns
            follower_count = random.randint(0, 50)  # Low followers
            following_count = random.randint(100, 2000)  # High following
            post_count = random.randint(0, 10)  # Low posts
            
            # Generate recent account creation
            account_age = random.randint(1, 30)  # Recent accounts
            
            account = {
                'id': i + 1,
                'username': username,
                'email': self.fake.email(),
                'bio': bio,
                'created_at': (datetime.now() - timedelta(days=account_age)).isoformat(),
                'follower_count': follower_count,
                'following_count': following_count,
                'post_count': post_count,
                'is_fake': True,
                'fake_score': random.uniform(0.7, 1.0)
            }
            
            fake_accounts.append(account)
        
        return fake_accounts
    
    def generate_real_accounts(self, count=1000):
        """Generate realistic account data with normal patterns"""
        real_accounts = []
        
        for i in range(count):
            # Generate realistic username patterns
            username_patterns = [
                lambda: self.fake.user_name(),
                lambda: f"{self.fake.first_name()}{self.fake.last_name()}",
                lambda: f"{self.fake.first_name()}{random.choice(['', '.', '_'])}{random.randint(1, 99)}",
                lambda: f"{self.fake.first_name().lower()}{random.choice(['', '1', '2', '3'])}",
                lambda: f"{self.fake.last_name().lower()}{random.choice(['', '1', '2', '3'])}",
                lambda: f"{self.fake.word()}{random.choice(['', '1', '2', '3'])}"
            ]
            
            username = random.choice(username_patterns)()
            
            # Generate realistic bio patterns
            bio_patterns = [
                f"Hi, I'm {self.fake.first_name()}! {self.fake.sentence()}",
                f"Living life one day at a time ✨",
                f"Passionate about {self.fake.word()} and {self.fake.word()}",
                f"📍 {self.fake.city()}, {self.fake.country()}",
                f"Student at {self.fake.company()}",
                f"Working at {self.fake.company()}",
                f"Love {self.fake.word()}, {self.fake.word()}, and {self.fake.word()}",
                f"Exploring the world 🌍",
                f"Photography enthusiast 📸",
                f"Food lover 🍕",
                f"Travel addict ✈️",
                f"Bookworm 📚",
                f"Fitness enthusiast 💪",
                f"Music lover 🎵",
                f"Tech geek 💻",
                f"Nature lover 🌿",
                f"Dog person 🐕",
                f"Cat person 🐱",
                f"Coffee addict ☕",
                f"Adventure seeker 🏔️"
            ]
            
            bio = random.choice(bio_patterns)
            
            # Generate realistic network patterns
            follower_count = random.randint(10, 500)
            following_count = random.randint(10, 300)
            post_count = random.randint(5, 100)
            
            # Generate varied account ages
            account_age = random.randint(30, 1000)  # Older accounts
            
            account = {
                'id': i + 1001,
                'username': username,
                'email': self.fake.email(),
                'bio': bio,
                'created_at': (datetime.now() - timedelta(days=account_age)).isoformat(),
                'follower_count': follower_count,
                'following_count': following_count,
                'post_count': post_count,
                'is_fake': False,
                'fake_score': random.uniform(0.0, 0.3)
            }
            
            real_accounts.append(account)
        
        return real_accounts
    
    def generate_training_data(self, fake_count=1000, real_count=1000):
        """Generate complete training dataset"""
        print(f"Generating {fake_count} fake accounts...")
        fake_accounts = self.generate_fake_accounts(fake_count)
        
        print(f"Generating {real_count} real accounts...")
        real_accounts = self.generate_real_accounts(real_count)
        
        # Combine datasets
        all_accounts = fake_accounts + real_accounts
        random.shuffle(all_accounts)
        
        # Save to JSON files
        with open('data/fake_accounts.json', 'w') as f:
            json.dump(fake_accounts, f, indent=2)
        
        with open('data/real_accounts.json', 'w') as f:
            json.dump(real_accounts, f, indent=2)
        
        # Create CSV for training
        df = pd.DataFrame(all_accounts)
        df.to_csv('data/training_data.csv', index=False)
        
        print(f"Generated {len(fake_accounts)} fake accounts and {len(real_accounts)} real accounts")
        print("Data saved to data/ directory")
        
        return all_accounts
    
    def generate_test_data(self, count=100):
        """Generate test data for evaluation"""
        test_accounts = []
        
        for i in range(count):
            is_fake = random.choice([True, False])
            
            if is_fake:
                # Generate fake account
                username = f"{self.fake.user_name()}{random.randint(1000, 9999)}"
                bio = f"💰 Make money fast! Click here: {self.fake.url()}"
                follower_count = random.randint(0, 20)
                following_count = random.randint(200, 1500)
                post_count = random.randint(0, 5)
                account_age = random.randint(1, 15)
                fake_score = random.uniform(0.7, 1.0)
            else:
                # Generate real account
                username = self.fake.user_name()
                bio = f"Hi, I'm {self.fake.first_name()}! {self.fake.sentence()}"
                follower_count = random.randint(20, 400)
                following_count = random.randint(20, 250)
                post_count = random.randint(10, 80)
                account_age = random.randint(100, 800)
                fake_score = random.uniform(0.0, 0.3)
            
            account = {
                'id': i + 2001,
                'username': username,
                'email': self.fake.email(),
                'bio': bio,
                'created_at': (datetime.now() - timedelta(days=account_age)).isoformat(),
                'follower_count': follower_count,
                'following_count': following_count,
                'post_count': post_count,
                'is_fake': is_fake,
                'fake_score': fake_score
            }
            
            test_accounts.append(account)
        
        return test_accounts

if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_training_data(1000, 1000) 