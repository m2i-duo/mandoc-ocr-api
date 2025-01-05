from fastapi import APIRouter, UploadFile, File
from starlette.responses import JSONResponse
from app.services.imageService import ImageService


# Initialize the ImageService
image_service = ImageService()


# Create a router with a prefix
crnnRouter = APIRouter()


@crnnRouter.post("/chunks")
async def chunks(image: UploadFile = File(...)):
    """
    Endpoint to process an image and return recognized words as chunks.
    """
    try:
        contents = await image.read()
        results = image_service.process_image(contents)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

@crnnRouter.post("/merged")
async def merged(image: UploadFile = File(...)):
    """
    Endpoint to process an image and return all recognized words as a single string.
    """
    try:
        contents = await image.read()
        results = image_service.process_image(contents)
        merged_text = " ".join([result["label"] for result in results if "label" in result])
        return {"text": merged_text}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
