"""
Main FastAPI application for VibeML backend.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from routers import train, predict, export, data_import
from models.experiment_store import get_db, engine, Base

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VibeML API",
    description="Machine Learning platform for model training, prediction, and experiment tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
@app.on_event("startup")
async def startup_event():
    """Initialize database and other startup tasks."""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize Redis connection if needed
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            logger.info(f"Redis configured at: {redis_url}")
        
        logger.info("VibeML API started successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup tasks on shutdown."""
    logger.info("VibeML API shutting down")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        
        return {
            "status": "healthy",
            "service": "VibeML API",
            "version": "1.0.0",
            "database": "connected",
            "timestamp": "2024-01-01T00:00:00Z"  # Will be replaced with actual timestamp
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to VibeML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "training": "/train",
            "data_import": "/import", 
            "prediction": "/predict",
            "export": "/export"
        }
    }


# API Info endpoint
@app.get("/api/info")
async def api_info():
    """Get API configuration and status."""
    return {
        "api_version": "1.0.0",
        "environment": os.getenv("DEBUG", "False"),
        "database_type": "sqlite" if "sqlite" in os.getenv("DATABASE_URL", "") else "postgresql",
        "redis_enabled": bool(os.getenv("REDIS_URL")),
        "celery_enabled": bool(os.getenv("CELERY_BROKER_URL")),
        "available_algorithms": [
            "linear_regression",
            "logistic_regression", 
            "random_forest",
            "gradient_boosting",
            "svm",
            "knn",
            "decision_tree",
            "naive_bayes",
            "neural_network",
            "xgboost",
            "lightgbm"
        ],
        "supported_file_formats": ["csv", "xlsx", "xls"],
        "max_dataset_size_mb": int(os.getenv("MAX_DATASET_SIZE_MB", 100))
    }


# Include routers
app.include_router(train.router, prefix="/api")
app.include_router(data_import.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(export.router, prefix="/api")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error in {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "path": str(request.url)
        }
    )


# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests."""
    start_time = None  # Placeholder for timing
    
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"Response: {response.status_code} for {request.method} {request.url}")
    
    return response


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting VibeML API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
