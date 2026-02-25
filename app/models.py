from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timezone


class MessageRequest(BaseModel):
    """Request model for message analysis"""
    text: str = Field(..., description="SMS message text to analyze", min_length=1)
    message_id: Optional[str] = Field(None, description="Optional message identifier")


class ScoreResponse(BaseModel):
    """Response model for spam scoring endpoint"""
    risk_score: float = Field(..., description="Risk score between 0 and 1")
    is_threat: bool = Field(..., description="Whether message is classified as spam/threat")
    model_version: str = Field(..., description="Version of the model used")
    message_id: Optional[str] = Field(None, description="Message identifier if provided")



class FeedbackRequest(BaseModel):
    """Request model for feedback collection"""
    message_id: str = Field(..., description="Message identifier")
    text: str = Field(..., description="Original message text")
    true_label: str = Field(..., description="Correct label: 'spam' or 'ham'")
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether models are loaded successfully")
    api_version: str = Field(..., description="API version")
    model_version: str = Field(..., description="Loaded model filename")
    title: str = Field(..., description="API title")
    description: str = Field(..., description="API description")
