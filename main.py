import numpy as np
import cv2
import gradio as gr
from ultralytics import YOLO
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv


print("Loading environment variables...")
load_dotenv()

HF_REPO_ID = "tententgc/Iskyn"
MODEL_FILENAME = "best.onnx"

print(f"Downloading '{MODEL_FILENAME}' from '{HF_REPO_ID}'...")

model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_FILENAME,
    token=os.getenv("HF_TOKEN") 
)

print(f"Model downloaded to: {model_path}")



print("Loading YOLO model...")
onnx_model = YOLO(model_path) 
print("Model loaded successfully.")



def predict_image(image_filepath, conf_threshold, iou_threshold):
    results = onnx_model.predict(
        image_filepath,
        conf=conf_threshold,
        iou=iou_threshold
    )
    result = results[0]
    im_array = result.plot()
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    return im_rgb

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence Threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU Threshold")
    ],
    outputs=gr.Image(type="numpy", label="Result"),
    title="Detection Face Skin",
    description="Upload an image and adjust the thresholds to fine-tune detection." 
)

iface.launch()