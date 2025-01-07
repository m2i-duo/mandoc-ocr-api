import os
import cv2
import numpy as np
import pytesseract
import base64
from pathlib import Path
import uuid
from app.utils.segmentImage import segment_image

class ArabicTextRecognitionService:
    def __init__(self):
        # Set the environment variable for Tesseract
        os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract/tessdata'

        # Set the path to Tesseract executable if not in PATH
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

        self.lang = 'ara'

    def clean_text(self, text):
        """
        Cleans the recognized Arabic text by removing unwanted characters and normalizing the text.
        """
        cleaned = text.strip()
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c.isspace() or c in ['؟', '،', '.'])  # Allow Arabic punctuation
        cleaned = ' '.join(cleaned.split())  # Normalize multiple spaces to a single space
        return cleaned


    def recognize_text(self, file_contents, type='chunks'):
        """
        Recognizes Arabic text from an image and returns a list of words with their corresponding images in base64 format.
        """
        try:
            # Decode the image from file contents
            nparr = np.frombuffer(file_contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Use grayscale directly

            # Segment the image into individual words
            if(type == 'chunks'):
                word_images = segment_image(img)
            else:
                word_images = [img]

            results = []
            for i, word_img in enumerate(word_images):
                # Save preprocessed image temporarily
                unique_name = f"word_{uuid.uuid4().hex}.png"
                unique_path = Path('/tmp', unique_name)
                cv2.imwrite(str(unique_path), word_img)

                # Recognize text using Tesseract
                custom_config = f'--oem 3 --psm 6 -l {self.lang}'
                recognized_text = pytesseract.image_to_string(word_img, config=custom_config)

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
            print(f"Error during text recognition: {e}")
            return {"status": "error", "message": str(e)}


