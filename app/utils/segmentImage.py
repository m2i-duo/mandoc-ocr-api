import cv2

def segment_image(image):
     word_images = []

     try:
         # Apply adaptive thresholding (invert to make text white on black background)
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

         # Sort bounding boxes within each row by their **right-to-left** horizontal position
         for row in rows:
             row.sort(key=lambda b: -b[0])  # Sort by x-coordinate in descending order (right-to-left)

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

     return word_images