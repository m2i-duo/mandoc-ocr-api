from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from app.services.tesseractService import ArabicTextRecognitionService

tesseractRouter = APIRouter()

service = ArabicTextRecognitionService()

@tesseractRouter.post("/chunks")
async def recognize_arabic_text(image: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and recognize Arabic text.
    """
    try:
        # Check file type
        # Read file contents
        image_contents = await image.read()
        # Recognize Arabic text from the image
        response = service.recognize_text(image_contents)
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@tesseractRouter.post("/merged")
async def recognize_arabic_text(image: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and recognize Arabic text.
    """
    try:
        # Read file contents
        image_contents = await image.read()
        # Recognize Arabic text from the image
        results = service.recognize_text(image_contents, type='merged')
        merged_text = " ".join([result["label"] for result in results if "label" in result])
        return {"text": merged_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))