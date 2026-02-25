import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import json
import uuid
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

# Import model class from training script
sys.path.append(str(Path(__file__).parent.parent))
from training.train_enhanced import EnhancedSpamPipeline

class ModelService:
    """Service for loading and using the enhanced ML model"""
    
    def __init__(self):
        self.enhanced_model = None
        self.feature_names = None
        self.loaded_model_filename = None
        self._load_model()
    
    def _load_model(self):
        """Load the enhanced model"""
        try:
            # Load enhanced model components
            enhanced_path = Path(__file__).parent.parent / "artifacts" / "enhanced_model.joblib"
            if enhanced_path.exists():
                model_package = joblib.load(enhanced_path)
                
                # Load model components and metadata
                self.tfidf_vectorizer = model_package['tfidf_vectorizer']
                self.classifier = model_package['classifier']
                self.feature_names = model_package['metadata']['feature_names']
                self.loaded_model_filename = "enhanced_model.joblib"
                
                logger.info(f"Loaded enhanced model components from {enhanced_path}")
                logger.info(f"Feature names: {self.feature_names}")
            else:
                logger.error(f"Enhanced model not found at {enhanced_path}")
                raise FileNotFoundError(f"Enhanced model not found at {enhanced_path}")
                
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            raise
    
    def predict_spam(self, text: str, heuristic_features: Dict[str, bool]) -> Tuple[float, bool]:
        """
        Predict spam probability for a text message using enhanced model
        
        Args:
            text: Message text to analyze
            heuristic_features: Pre-computed heuristic features (required)
            
        Returns:
            Tuple of (risk_score, is_threat)
        """
        try:
            if not hasattr(self, 'tfidf_vectorizer') or not hasattr(self, 'classifier'):
                raise ValueError("Enhanced model components not available")
            
            # Transform text to TF-IDF features
            tfidf_features = self.tfidf_vectorizer.transform([text])
            
            # Convert heuristic features dict to array
            heuristic_array = np.array([[int(heuristic_features[name]) for name in self.feature_names]], dtype=np.float32)
            
            # Combine TF-IDF and heuristic features
            from scipy.sparse import hstack
            combined_features = hstack([tfidf_features, heuristic_array])
            
            # Get prediction probabilities from classifier
            probabilities = self.classifier.predict_proba(combined_features)[0]
            
            # Get spam probability (class 1)
            risk_score = float(probabilities[1])
            
            # Classify as threat if risk_score > 0.5
            is_threat = risk_score > 0.5
            
            return risk_score, is_threat
            
        except Exception as e:
            logger.error(f"Error in spam prediction: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """Check if service is healthy (enhanced model components loaded)"""
        return hasattr(self, 'tfidf_vectorizer') and hasattr(self, 'classifier')


class FeedbackService:
    """Service for handling feedback storage"""
    
    def __init__(self):
        self.feedback_file = Path(__file__).parent.parent / "data" / "feedback.jsonl"
        self.feedback_file.parent.mkdir(exist_ok=True)
    
    def store_feedback(self, message_id: str, text: str, true_label: str) -> str:
        """
        Store feedback for model improvement
        
        Args:
            message_id: Unique message identifier
            text: Original message text
            true_label: Correct label ('spam' or 'ham')
            
        Returns:
            Feedback record ID
        """
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback_record = {
                "feedback_id": feedback_id,
                "message_id": message_id,
                "text": text,
                "true_label": true_label,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Append to JSONL file
            with open(self.feedback_file, "a") as f:
                f.write(json.dumps(feedback_record) + "\n")
            
            logger.info(f"Stored feedback record {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            raise


# Global service instances
model_service = ModelService()
feedback_service = FeedbackService()
