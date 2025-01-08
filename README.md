# Mandoc OCR Docker Deployment

Mandoc OCR is an Arabic Optical Character Recognition (OCR) system utilizing Recurrent Neural Networks (RNN) and Connectionist Temporal Classification (CTC) for accurate text recognition.

This README provides instructions for setting up and running the Mandoc OCR API and UI using Docker.

## Prerequisites
Ensure the following are installed on your system:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/) (optional, for future integration)

## Commands Overview

### 1. Pull the Docker Images
Retrieve the latest images for both the API and the UI:

```bash
docker pull ghcr.io/m2i-duo/mandoc-ocr-api:latest
docker pull ghcr.io/m2i-duo/mandoc-ocr-ui/web:latest
```

### 2. Run the OCR API
Start the OCR API on port `8000`:

```bash
docker run -p 8000:8000 ghcr.io/m2i-duo/mandoc-ocr-api
```

### 3. Run the OCR UI
Start the OCR UI on port `3000`:

```bash
docker run -p 3000:3000 ghcr.io/m2i-duo/mandoc-ocr-ui/web
```

## Access the Services

### OCR API
Once the container is running, the OCR API will be accessible at:

```
http://localhost:8000
```

### OCR UI
Once the container is running, the OCR UI will be accessible at:

```
http://localhost:3000
```

## Notes
- Ensure that ports `8000` and `3000` are available on your machine.
- Use `docker ps` to verify the containers are running.
- To stop a container, use `docker stop <container_id>`.
- For persistent data, consider mounting volumes (not covered in this README).

## Troubleshooting
- If a port conflict occurs, ensure no other service is using the same port or modify the `-p` option to use different ports (e.g., `-p 8080:8000`).
- Check logs for more details using `docker logs <container_id>`.

## Additional Resources
- [Docker Documentation](https://docs.docker.com/)
- Contact the project maintainers for support or issues.

## Acknowledgements
This project builds upon the work and insights from the [Arabic Deep Learning OCR](https://github.com/msfasha/Arabic-Deep-Learning-OCR) repository. Special thanks to the contributors for their foundational research and implementation.

