from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from ultralytics import YOLO

print("載入BLIP...")
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
print("載入CLIP...")
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)
print("載入YOLO...")
YOLO_model = YOLO("yolo11x.pt")

print("models_load:載入完成")