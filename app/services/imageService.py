import cv2
import numpy as np
import base64
from app.models.crnn_ctc_model.Main import inferSingleImage
from app.models.crnn_ctc_model.Model import Model
from app.utils.segmentImage import segment_image
from app.utils.sample_preprocessing import preprocess  # Import the preprocess function
from pathlib import Path
import uuid
from config import config

class ImageService:
    def __init__(self):
        self.model = Model(config.DECODER_TYPE, mustRestore=True, dump=False)

    def clean_text(self ,text):
        """
        Cleans the recognized text by removing unwanted characters and normalizing the text.
        """
        # Define your cleaning rules here
        cleaned = text.strip()  # Remove leading/trailing whitespace
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c.isspace())  # Keep alphanumeric and spaces
        cleaned = ' '.join(cleaned.split())  # Normalize multiple spaces to a single space
        return cleaned

    def process_image(self, file_contents):
        """
        Process an image containing handwritten text and segment it into individual words.
        Each word is recognized and returned along with its corresponding image in base64 format.
        """
        try:
            # Decode the image from file contents
            nparr = np.frombuffer(file_contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Use grayscale directly

            # Debugging: Check pixel value range
            print(f"Original image pixel range: {img.min()} - {img.max()}")

            # Segment the image into individual words
            word_images = segment_image(img)

            results = []
            for i, word_img in enumerate(word_images):
                # Save preprocessed image temporarily
                unique_name = f"word_{uuid.uuid4().hex}.png"
                unique_path = Path(config.TEMP_PATH, unique_name)
                cv2.imwrite(str(unique_path), word_img)

                # Recognize text
                recognized_text = inferSingleImage(self.model, str(unique_path))

                cleaned_text = self.clean_text(recognized_text)

                # Encode the original word image to base64
                _, buffer = cv2.imencode('.png', word_img)
                img_str = base64.b64encode(buffer).decode('utf-8')

                # Append the result with recognized text and word image
                results.append({"label": cleaned_text, "image": img_str})

                # Cleanup temporary file
                unique_path.unlink()

            return results

        except Exception as e:
            print(f"Error during image processing: {e}")
            return [{"label": "", "image": "", "error": str(e)}]

