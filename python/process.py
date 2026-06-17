# 先設定:
# 終端機輸入 py -m pip install ultralytics transformers pillow torch opencv-python flask
# 按 Ctrl + Shift + P  
# 輸入 Python: Select Interpreter 然後找能用的python版本
# 在vscode看.db:裝"SQLite Viewer"外掛/插件

# 沒事別import 因preset那邊要跑很久

#==Import==
#別人的庫
from PIL import Image
import cv2
import os
import json
#我做的庫
import db
from models_load import clip_processor, clip_model, blip_processor, blip_model, YOLO_model

#==Preset==
BASEPATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEPATH)
db.create_table()

#勿隨意改
LABELS = [
    "hand",
    "electronic device",
    "stationery item",
    "book or document",
    "storage box",
    "cup or bowl or plate",
    "food or drink",
    "anime figure",
    "plush toy",
    "money",
    "cable or charger",
    "wearable item",
    "random object"
]


#==define==
#輸入圖片 理論上是點陣圖都能用 但我沒試過 最好是用.jpg
def input_img(info:dict, name:str, folder_path=None):
    if len(name.split(".")) == 1 :
        name = name+".jpg"
        print("檔名未加 \".***\", 已自動加上\".jpg\"")

    if folder_path is None:
        folder_path = os.path.join(BASEPATH,"images_esp_upload")
    path = os.path.join(folder_path,name)

    print(f"輸入圖片:  name:{name}, path:{path}")
    info["img_name"] = name
    info["img_path"] = path
    

def check_img_path(info:dict):
    if "img_path" not in info:
        raise ValueError("info[\"img_path\"]不存在")
    if not os.path.exists(info["img_path"]):
        raise FileNotFoundError(f"info[\"img_path\"] {info['img_path']} 沒圖片")
    
def check_crops(info:dict):
    if "crops" not in info:
        raise ValueError("info[\"crops\"]不存在")
    if not isinstance(info["crops"], list):
        raise TypeError('info["crops"]不是list')
    
#YOLO找框框
def YOLO_find_crops(info:dict):
    check_img_path(info)

    results = YOLO_model(info["img_path"], verbose=False)
    result = results[0]

    info["crops"] = []
    for i,box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        info["crops"].append({"crop_position":{'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2},
                              "YOLO_confidence":conf})
        

#clip
def CLIP_describe_and_embedding(info:dict, image):
    check_img_path(info)
    check_crops(info)

    for crop_info in info["crops"]:
        pos = crop_info["crop_position"]
        x1, y1, x2, y2 = pos["x1"], pos["y1"], pos["x2"], pos["y2"] 
        crop_image = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))

        inputs = clip_processor(
            text=LABELS,
            images=crop_pil,
            return_tensors="pt",
            padding=True    
        )
        outputs = clip_model(**inputs)

        logits = outputs.logits_per_image
        best = logits.argmax().item()

        image_features = outputs.image_embeds

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        embedding_list = image_features[0].tolist()
        
        crop_info["CLIP_embedding"] = embedding_list
        crop_info["CLIP_describe"] = LABELS[best]
    
#blip 從for i到crop_pil都跟clip一樣
def BLIP_describe(info:dict, image):
    check_img_path(info)
    check_crops(info)

    for crop_info in info["crops"]:
        pos = crop_info["crop_position"]
        x1, y1, x2, y2 = pos["x1"], pos["y1"], pos["x2"], pos["y2"] 
        crop_image = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))

        blip_inputs = blip_processor(
            images=crop_pil,
            return_tensors="pt"
        )

        blip_outputs = blip_model.generate(**blip_inputs)

        blip_caption = blip_processor.decode(
            blip_outputs[0],
            skip_special_tokens=True
        )
        crop_info["BLIP_describe"] = blip_caption

#存成json
def save_as_json(info:dict, folder:str="JSON_results_folder"):
    check_img_path(info)

    os.makedirs(folder, exist_ok=True)

    json_name = info["img_name"].split(".")[0]+".json"
    json_path = os.path.join(folder, json_name)

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            info,
            f,
            ensure_ascii=False,
            indent=4
        )

#組合
def process_img(img_name:str, folder_path=None):
    info = {}

    input_img(info, img_name, folder_path)

    image = cv2.imread(info["img_path"])
    if image is None:
        raise FileNotFoundError(info["img_path"])
    
    print(f"開始辨識{info['img_name']}")

    YOLO_find_crops(info)

    if info["crops"]:
        CLIP_describe_and_embedding(info, image)
        BLIP_describe(info, image)
        print("完成")
    else:
        print("沒東西")

    save_as_json(info)
    db.save_to_db(info)

    return info

#==main script==
# if __name__ == "__main__":
#     process_img("30.jpg")
#     print("完成")

if __name__ == "__main__":
    dir = "test_images"
    for file_name in os.listdir(dir):
        if file_name.endswith(".jpg"):
            process_img(img_name=file_name, folder_path=dir)