from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

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
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
