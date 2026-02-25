import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import patch

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns correct structure"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data
        assert "model_version" in data
        assert "title" in data
        assert "description" in data
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestScoreEndpoint:
    """Test spam scoring endpoint"""
    
    def test_score_ham_message(self):
        """Test scoring a ham (non-spam) message"""
        request_data = {
            "text": "Hey, how are you doing today?",
            "message_id": "test_ham_1"
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "risk_score" in data
        assert "is_threat" in data
        assert "model_version" in data
        assert "message_id" in data
        assert 0 <= data["risk_score"] <= 1
        assert data["is_threat"] == False  # Should classify as ham (not threat)
    
    def test_score_spam_message(self):
        """Test scoring a spam message"""
        request_data = {
            "text": "URGENT! Click here to claim your £1000 prize now! Limited time offer!",
            "message_id": "test_spam_1"
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "risk_score" in data
        assert "is_threat" in data
        assert "message_id" in data
        assert 0 <= data["risk_score"] <= 1
        assert data["is_threat"] == True  # Should classify as spam (threat)
    
    def test_score_empty_text(self):
        """Test scoring with empty text should fail"""
        request_data = {
            "text": "",
            "message_id": "test_empty"
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_score_without_message_id(self):
        """Test scoring without message_id (optional field)"""
        request_data = {
            "text": "Hello world"
        }
        
        response = client.post("/score", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "message_id" in data
    
    def test_score_missing_text(self):
        """Test scoring with missing required text field"""
        response = client.post("/score", json={"message_id": "test"})
        assert response.status_code == 422
    
    def test_score_invalid_json(self):
        """Test scoring with invalid JSON"""
        response = client.post("/score", content="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422




class TestFeedbackEndpoint:
    """Test feedback submission endpoint"""
    
    @patch('app.services.feedback_service.store_feedback')
    def test_submit_valid_feedback(self, mock_store):
        """Test submitting valid feedback"""
        mock_store.return_value = "test-feedback-id"
        
        request_data = {
            "message_id": "test_msg_123",
            "text": "Test message for feedback",
            "true_label": "spam"
        }
        
        response = client.post("/feedback", json=request_data)
        assert response.status_code == 204  # No Content
        assert response.text == ""  # No response body
        mock_store.assert_called_once()
    
    @patch('app.services.feedback_service.store_feedback')
    def test_submit_feedback_ham_label(self, mock_store):
        """Test submitting feedback with ham label"""
        mock_store.return_value = "test-feedback-id"
        
        request_data = {
            "message_id": "test_msg_456",
            "text": "Test ham message",
            "true_label": "ham"
        }
        
        response = client.post("/feedback", json=request_data)
        assert response.status_code == 204  # No Content
        assert response.text == ""  # No response body
        mock_store.assert_called_once()
    
    def test_submit_invalid_label(self):
        """Test submitting feedback with invalid label"""
        request_data = {
            "message_id": "test_msg_789",
            "text": "Test message",
            "true_label": "invalid_label"
        }
        
        response = client.post("/feedback", json=request_data)
        assert response.status_code == 400  # Bad request
    
    def test_feedback_missing_message_id(self):
        """Test feedback with missing required message_id field"""
        response = client.post("/feedback", json={
            "text": "test",
            "true_label": "spam"
        })
        assert response.status_code == 422
    
