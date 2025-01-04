# Use a base image with Python 3.7
FROM python:3.7-slim

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the Docker image
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the Docker image
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]