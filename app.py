# app.py
import io
import uvicorn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
import cv2
from huggingface_hub import hf_hub_download
import os
import uuid

# --- FastAPI and Template Setup ---
app = FastAPI(title="YOLOv8 ONNX Object Detection Demo")

# Mount a static directory to serve saved images
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# --- Model Loading and Configuration ---
# Download the ONNX model file and get its path
try:
    onnx_model_path = hf_hub_download(repo_id="tententgc/Iskyn", filename="best.onnx")
    session = ort.InferenceSession(onnx_model_path)
    print("ONNX model loaded successfully.")
except Exception as e:
    print(f"Failed to load ONNX model: {e}")
    session = None

if session:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape[2:]  # Get the expected image size
else:
    input_name = None
    output_names = []
    input_shape = (640, 640)  # Default size if model fails to load

# Define the class names for your model
# IMPORTANT: Update this with the actual class names your model was trained on
CLASSES = [
    "melasma", "acne", "wrinkle" 
]

# A dictionary to map class names to colors for plotting
COLORS = {
    "melasma": "red",
    "acne": "green",
    "wrinkle": "blue",
    # Add more classes and colors as needed
}

# --- Helper Functions ---
def preprocess_image(image: Image.Image, size: tuple) -> np.ndarray:
    """Preprocesses an image for model inference."""
    image = image.resize(size)
    image = np.array(image)
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.expand_dims(image, axis=0) # Add batch dimension
    image = image.astype(np.float32) / 255.0  # Normalize
    return image

def postprocess_output(output, original_size, input_shape, conf_threshold=0.25, iou_threshold=0.45):
    """Post-processes the model output to get bounding boxes, scores, and class IDs."""
    output = np.squeeze(output).T
    scores = np.max(output[:, 4:], axis=1)
    filtered_indices = scores > conf_threshold
    output = output[filtered_indices]
    scores = scores[filtered_indices]

    if not len(output):
        return []

    boxes = output[:, :4]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    class_ids = np.argmax(output[:, 4:], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes.astype(np.int32), scores.astype(np.float32), conf_threshold, iou_threshold)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x1, y1, x2, y2 = box.astype(int)
            class_id = class_ids[i]
            score = scores[i]

            original_width, original_height = original_size
            resized_width, resized_height = input_shape
            x1 = int(x1 * original_width / resized_width)
            y1 = int(y1 * original_height / resized_height)
            x2 = int(x2 * original_width / resized_width)
            y2 = int(y2 * original_height / resized_height)

            detections.append({
                "class_name": CLASSES[class_id],
                "confidence": float(score),
                "box": [x1, y1, x2, y2]
            })
    return detections

def draw_boxes_on_image(image, detections):
    """Draws bounding boxes, class names, and confidence scores on an image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
        print("Arial font not found, using default font.")

    for detection in detections:
        box = detection['box']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        color = COLORS.get(class_name, "white")
        draw.rectangle(box, outline=color, width=3)
        
        label = f"{class_name}: {confidence:.2f}"
        
        # Use textbbox() to get text dimensions
        text_x, text_y, text_width, text_height = draw.textbbox((0, 0), label, font=font)
        
        # Position text slightly above the top-left corner
        text_position_y = box[1] - text_height - 5
        if text_position_y < 0:
            text_position_y = box[1] + 5 # Draw below if not enough space above
            
        draw.rectangle([box[0], text_position_y, box[0] + text_width, text_position_y + text_height], fill=color)
        draw.text((box[0], text_position_y), label, fill="black", font=font)
    return image

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "error_message": None})

@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(request: Request, file: UploadFile = File(...)):
    """Handle image upload, run detection, and return plotted image."""
    if not session:
        return templates.TemplateResponse("index.html", {"request": request, "error_message": "ONNX model not loaded."})
    
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse("index.html", {"request": request, "error_message": "Invalid file type. Please upload an image."})

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size

        # Preprocess, run inference, and post-process
        preprocessed_image = preprocess_image(image, size=input_shape)
        outputs = session.run(output_names, {input_name: preprocessed_image})
        detections = postprocess_output(outputs, original_size, input_shape)

        # Draw boxes on the original image
        plotted_image = draw_boxes_on_image(image.copy(), detections)

        # Create a unique filename and save the plotted image
        unique_filename = f"{uuid.uuid4()}.jpg"
        output_image_path = os.path.join("static", "output", unique_filename)
        plotted_image.save(output_image_path)
        
        image_url = f"/static/output/{unique_filename}"
        
        return templates.TemplateResponse("index.html", {"request": request, "image_url": image_url})
    
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error_message": f"An error occurred: {e}"})

if __name__ == "__main__":
    # Create the static/output directory if it doesn't exist
    os.makedirs(os.path.join("static", "output"), exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
