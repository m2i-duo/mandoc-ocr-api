import os
import cv2
import numpy as np
import pytesseract
import base64
from pathlib import Path
import uuid

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

    def recognize_text(self, file_contents):
        """
        Recognizes Arabic text from an image and returns a list of words with their corresponding images in base64 format.
        """
        try:
            # Decode the image from file contents
            nparr = np.frombuffer(file_contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Use grayscale directly

            # Segment the image into individual words
            word_images = self.segment_image(img)

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

    def segment_image(self, image):
        word_images = []

        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Apply morphological operations to merge close components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Perform connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

            # Extract bounding boxes, filtering out small components
            bounding_boxes = [
                (x, y, w, h) for x, y, w, h, area in stats[1:] if area >= 50
            ]  # Skip label 0 (background)

            # Group bounding boxes into rows
            rows = []
            for box in bounding_boxes:
                x, y, w, h = box
                mid_y = y + h // 2
                added_to_row = False

                # Check if the box belongs to an existing row
                for row in rows:
                    if abs(row[0][1] + row[0][3] // 2 - mid_y) <= h // 2:  # Adjust threshold as needed
                        row.append(box)
                        added_to_row = True
                        break

                # If not added to any row, create a new row
                if not added_to_row:
                    rows.append([box])

            # Sort rows by their vertical position
            rows.sort(key=lambda r: r[0][1])

            # Within each row, sort boxes by horizontal position
            for row in rows:
                row.sort(key=lambda b: b[0])

                # Extract and pad word images
                for x, y, w, h in row:
                    cropped_image = image[y:y + h, x:x + w]
                    padding = 8  # Adjust padding size as needed
                    padded_image = cv2.copyMakeBorder(
                        cropped_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255
                    )
                    word_images.append(padded_image)

        except Exception as e:
            print(f"Error during segmentation: {e}")

        return word_images[::-1]