from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import sys

# API Version
API_VERSION = "1.0.0"

from .models import (
    MessageRequest, ScoreResponse, 
    FeedbackRequest, HealthResponse
)
from .services import model_service, feedback_service

# Import feature extraction directly
sys.path.append(str(Path(__file__).parent.parent))
from features.tagging import extract_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting SMS Spam Detection API")
    logger.info(f"Model service healthy: {model_service.is_healthy()}")
    yield
    # Shutdown (if needed)
    logger.info("Shutting down SMS Spam Detection API")


# Create FastAPI app
app = FastAPI(
    title="SMS Spam Detection API",
    description="Production-ready API for SMS spam detection with heuristic feature tagging",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WARNING: Allows all origins - restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_service.is_healthy() else "unhealthy",
        model_loaded=model_service.is_healthy(),
        api_version=API_VERSION,
        model_version=model_service.loaded_model_filename or "none",
        title=app.title,
        description=app.description
    )


@app.post("/score", response_model=ScoreResponse)
async def score_message(request: MessageRequest):
    """
    Score a message for spam probability
    
    Returns risk score and threat classification
    """
    try:
        logger.info(f"Scoring message: {request.message_id or 'anonymous'}")
        
        # Extract heuristic features once
        heuristic_features = extract_all_features(request.text)
        
        # Get prediction from model service (using enhanced model)
        risk_score, is_threat = model_service.predict_spam(
            text=request.text,
            heuristic_features=heuristic_features
        )
        
        response = ScoreResponse(
            risk_score=risk_score,
            is_threat=is_threat,
            model_version=model_service.loaded_model_filename or "unknown",
            message_id=request.message_id
        )
        
        # Log active features for monitoring and debugging
        active_features = [name for name, value in heuristic_features.items() if value]
        logger.info(f"Score result - Risk: {risk_score:.3f}, Threat: {is_threat}, Active features: {active_features}")
        return response
        
    except Exception as e:
        logger.error(f"Error scoring message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )



@app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for model improvement
    
    Stores feedback data for future model retraining
    Returns 204 No Content on success
    """
    try:
        logger.info(f"Receiving feedback for message: {request.message_id}")
        
        # Validate true_label
        if request.true_label.lower() not in ['spam', 'ham']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="true_label must be 'spam' or 'ham'"
            )
        
        # Store feedback
        feedback_id = feedback_service.store_feedback(
            message_id=request.message_id,
            text=request.text,
            true_label=request.true_label.lower()
        )
        
        logger.info(f"Feedback stored with ID: {feedback_id}")
        # Return 204 No Content (no response body)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing feedback: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": app.title,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /score": "Score message for spam probability",
            "POST /feedback": "Submit feedback for model improvement"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
