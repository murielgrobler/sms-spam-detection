"""
Heuristic tagging functions for SMS spam detection.

This module provides fast, deterministic feature extraction using regex patterns
and keyword matching. These features are designed to be interpretable and 
production-ready for security ML applications.
"""

import re
from typing import Dict, List


def extract_link_features(text: str) -> Dict[str, bool]:
    """
    Extract link-related features from text.
    
    Args:
        text: Input SMS message text
        
    Returns:
        Dictionary with link-related boolean features
    """
    text_lower = text.lower()
    
    # URL patterns
    url_patterns = [
        r'https?://\S+',           # http:// or https://
        r'www\.\S+',               # www.domain
        r'\S+\.(com|org|net|co\.uk|info|biz)\b',  # domain extensions
        r'bit\.ly/\S+',            # shortened URLs
        r'tinyurl\.com/\S+',       # tinyurl
        r'goo\.gl/\S+',            # Google shortened URLs
    ]
    
    has_url = any(re.search(pattern, text_lower) for pattern in url_patterns)
    
    # Link-related keywords
    link_keywords = ['click', 'visit', 'go to', 'check out', 'see more', 'link']
    has_link_keyword = any(keyword in text_lower for keyword in link_keywords)
    
    return {
        'link_present': has_url,
        'link_keyword': has_link_keyword,
        'link_any': has_url or has_link_keyword
    }


def extract_money_features(text: str) -> Dict[str, bool]:
    """
    Extract money/financial-related features from text.
    
    Args:
        text: Input SMS message text
        
    Returns:
        Dictionary with money-related boolean features
    """
    text_lower = text.lower()
    
    # Currency symbols and amounts
    money_patterns = [
        r'[£$€¥]\s*\d+',                    # Currency symbols with numbers
        r'\d+\s*(pounds?|dollars?|euros?|usd|gbp|eur)',  # Numbers with currency words
        r'\d+\s*p\b',                       # Pence (e.g., "50p")
        r'free\s+money',                    # "free money"
        r'\d+\s*(k|thousand|million)',      # Large amounts
    ]
    
    has_money_amount = any(re.search(pattern, text_lower) for pattern in money_patterns)
    
    # Financial keywords
    financial_keywords = [
        'prize', 'win', 'won', 'winner', 'cash', 'reward', 'bonus',
        'free', 'offer', 'deal', 'discount', 'save', 'cheap',
        'cost', 'price', 'pay', 'payment', 'charge', 'bill'
    ]
    
    has_financial_keyword = any(keyword in text_lower for keyword in financial_keywords)
    
    # Prize/lottery specific
    prize_keywords = ['lottery', 'jackpot', 'congratulations', 'selected', 'chosen']
    has_prize_keyword = any(keyword in text_lower for keyword in prize_keywords)
    
    return {
        'money_amount': has_money_amount,
        'financial_keyword': has_financial_keyword,
        'prize_keyword': has_prize_keyword,
        'money_any': has_money_amount or has_financial_keyword or has_prize_keyword
    }


def extract_urgency_features(text: str) -> Dict[str, bool]:
    """
    Extract urgency-related features from text.
    
    Args:
        text: Input SMS message text
        
    Returns:
        Dictionary with urgency-related boolean features
    """
    text_lower = text.lower()
    
    # Urgency keywords
    urgency_keywords = [
        'urgent', 'immediately', 'asap', 'now', 'quick', 'hurry',
        'expires', 'expiry', 'deadline', 'limited time', 'act fast',
        'don\'t delay', 'last chance', 'final notice', 'today only'
    ]
    
    has_urgency = any(keyword in text_lower for keyword in urgency_keywords)
    
    # Time-sensitive patterns
    time_patterns = [
        r'within\s+\d+\s+(hours?|days?|minutes?)',  # "within 24 hours"
        r'expires?\s+(today|tomorrow|soon)',         # "expires today"
        r'\d+\s+(hours?|days?)\s+(left|remaining)', # "24 hours left"
    ]
    
    has_time_pressure = any(re.search(pattern, text_lower) for pattern in time_patterns)
    
    # Exclamation marks (urgency indicator)
    exclamation_count = text.count('!')
    has_multiple_exclamations = exclamation_count >= 2
    
    return {
        'urgency_keyword': has_urgency,
        'time_pressure': has_time_pressure,
        'multiple_exclamations': has_multiple_exclamations,
        'urgency_any': has_urgency or has_time_pressure or has_multiple_exclamations
    }


def extract_account_features(text: str) -> Dict[str, bool]:
    """
    Extract account/security-related features from text.
    
    Args:
        text: Input SMS message text
        
    Returns:
        Dictionary with account-related boolean features
    """
    text_lower = text.lower()
    
    # Account action keywords
    account_actions = [
        'verify', 'confirm', 'update', 'activate', 'suspend', 'block',
        'login', 'log in', 'sign in', 'password', 'username', 'pin',
        'security', 'account', 'profile', 'details', 'information'
    ]
    
    has_account_action = any(keyword in text_lower for keyword in account_actions)
    
    # Security-related terms
    security_keywords = [
        'fraud', 'suspicious', 'unauthorized', 'breach', 'compromise',
        'protect', 'secure', 'safety', 'alert', 'warning'
    ]
    
    has_security_keyword = any(keyword in text_lower for keyword in security_keywords)
    
    # Banking/financial institutions
    bank_keywords = [
        'bank', 'paypal', 'visa', 'mastercard', 'amazon', 'ebay',
        'apple', 'google', 'microsoft', 'facebook', 'twitter'
    ]
    
    has_institution = any(keyword in text_lower for keyword in bank_keywords)
    
    # Phishing patterns
    phishing_patterns = [
        r'click\s+here\s+to\s+(verify|confirm|update)',
        r'your\s+(account|card|payment)\s+(has\s+been|is)\s+(suspended|blocked|compromised)',
        r'we\s+(need|require)\s+you\s+to\s+(verify|confirm|update)'
    ]
    
    has_phishing_pattern = any(re.search(pattern, text_lower) for pattern in phishing_patterns)
    
    return {
        'account_action': has_account_action,
        'security_keyword': has_security_keyword,
        'institution_mention': has_institution,
        'phishing_pattern': has_phishing_pattern,
        'account_any': has_account_action or has_security_keyword or has_institution or has_phishing_pattern
    }


def extract_all_features(text: str) -> Dict[str, bool]:
    """
    Extract all heuristic features from text.
    
    Args:
        text: Input SMS message text
        
    Returns:
        Dictionary with all extracted boolean features
    """
    features = {}
    
    # Extract all feature categories
    features.update(extract_link_features(text))
    features.update(extract_money_features(text))
    features.update(extract_urgency_features(text))
    features.update(extract_account_features(text))
    
    # Meta features
    features['text_length'] = len(text) > 100  # Long messages
    features['all_caps_words'] = len(re.findall(r'\b[A-Z]{3,}\b', text)) > 0  # ALL CAPS words
    features['phone_number'] = bool(re.search(r'\b\d{10,}\b', text))  # Phone numbers
    
    return features


def get_feature_names() -> List[str]:
    """
    Get list of all feature names that will be extracted.
    
    Returns:
        List of feature names
    """
    # Generate a sample to get all feature names
    sample_features = extract_all_features("sample text")
    return list(sample_features.keys())


if __name__ == "__main__":
    # Test the functions with sample messages
    spam_message = "URGENT! You've won £1000! Click here to verify your account: www.fake-bank.com"
    ham_message = "Hey, are you free for dinner tonight? Let me know!"
    
    print("=== SPAM MESSAGE FEATURES ===")
    spam_features = extract_all_features(spam_message)
    for feature, value in spam_features.items():
        if value:  # Only show True features
            print(f"{feature}: {value}")
    
    print("\n=== HAM MESSAGE FEATURES ===")
    ham_features = extract_all_features(ham_message)
    for feature, value in ham_features.items():
        if value:  # Only show True features
            print(f"{feature}: {value}")
    
    print(f"\nTotal features: {len(get_feature_names())}")
    print(f"Feature names: {get_feature_names()}")
