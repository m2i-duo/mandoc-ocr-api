import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.crnnRouter import crnnRouter
from app.routers.tesseractRouter import tesseractRouter

# Initialize FastAPI application
app = FastAPI()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins (e.g., ["http://example.com"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow specific HTTP methods if needed
    allow_headers=["*"],  # Allow specific headers if needed
)

@app.get("/")
async def root():
    """
    Root endpoint to check API status.
    """
    return {"message": "CRNN API is running"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    """
    Simple greeting endpoint.
    """
    return {"message": f"Hello {name}"}


# Include the router in the main application
app.include_router(crnnRouter, prefix="/api/v1/crnn")
app.include_router(tesseractRouter, prefix="/api/v1/tesseract")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
