import cv2
import numpy as np
import base64
from app.models.crnn_ctc_model.Main import inferSingleImage
from app.models.crnn_ctc_model.Model import Model
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
            word_images = self.segment_image(img)

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
