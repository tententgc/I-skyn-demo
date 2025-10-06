import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os

# The URL of your FastAPI predict endpoint
url = "http://127.0.0.1:8000/predict"


image_path = "acne-face-2-18.jpg" 
output_path = "result.jpg" 


COLORS = {
    "acne": "red",
    "melasma": "green",
    "wrinkle": "blue"
}

def draw_boxes_on_image(image, detections):
    """Draws bounding boxes, class names, and confidence scores on an image."""
    draw = ImageDraw.Draw(image)
    try:
        # Try to use a better-looking font if available
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        print("Arial font not found, using default font.")

    for detection in detections:
        box = detection['box']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get color based on class name, defaulting to a solid color if not found
        color = COLORS.get(class_name, "white")
        
        # Draw the rectangle
        draw.rectangle(box, outline=color, width=3)
        
        # Create the label text with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        
        # Use textbbox() to get text dimensions
        # It returns a tuple: (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Define text position slightly above the top-left corner of the box
        text_x = box[0]
        text_y = box[1] - text_height - 5  # 5 pixels padding
        
        # Ensure text is not drawn off the top of the image
        if text_y < 0:
            text_y = box[1] + 5 # Draw below the box if no space above
        
        # Draw a filled background for the text for better visibility
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)
        
        # Draw the label text
        draw.text((text_x, text_y), label, fill="black", font=font)
        
    return image

try:
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: The image file was not found at {image_path}")

    # Open the image file in binary mode
    with open(image_path, "rb") as f:
        files = {"file": f}
        
        # Send the POST request to the FastAPI endpoint
        response = requests.post(url, files=files)
        
    # Check for a successful response (status code 200)
    if response.status_code == 200:
        detections = response.json().get("detections", [])
        
        if detections:
            print("Detections found:", detections)
            # Load the original image again for plotting
            original_image = Image.open(image_path).convert("RGB")
            
            # Draw the detections on the image
            plotted_image = draw_boxes_on_image(original_image, detections)
            
            # Save the new image with the plots
            plotted_image.save(output_path)
            print(f"Success! Plotted image saved to: {output_path}")
            
        else:
            print("No objects were detected.")
            
    else:
        print(f"Error: API returned status code {response.status_code}")
        print("Response:", response.text)

except requests.exceptions.RequestException as e:
    print(f"An error occurred while connecting to the API: {e}")



# adding test copmmand 
