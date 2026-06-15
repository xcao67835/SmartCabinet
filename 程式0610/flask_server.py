import flask
from datetime import datetime
import os
import threading
import time
import cv2

#import models_load
import process
from process import BASEPATH
import search
import db
import esp_request


esp_ip = esp_request.esp_ip
esp_status = None

def update_esp_status(ip=None):
    global esp_status

    response = esp_request.request_esp_status(ip)

    if response is None:
        esp_status = None
        print(f"時間{datetime.now().strftime("%H:%M:%S")}, ESP連線狀態: 離線", end = ",  ")
        return

    esp_status = response.json()

def esp_threading(ip=None, density=None, led=None, sleep_time=None):
    if ip is None:ip = esp_ip
    if density is None:density = 11
    if led is None:led = 0
    if sleep_time is None:actual_sleep_time = 5
    else:actual_sleep_time = sleep_time

    global esp_status

    while True:
        update_esp_status(ip)
        

        if esp_status is None:
            print(f"約{round(actual_sleep_time+2)}秒後將重新檢測")
            time.sleep(actual_sleep_time)
            actual_sleep_time = actual_sleep_time*1.1
            continue

        if esp_status["framesize"] != density:
            esp_request.set_frame_size(val=density, ip=ip)
        if esp_status["led_intensity"] != led:
            esp_request.set_led_intensity(val=led, ip=ip)

        actual_sleep_time = sleep_time
        time.sleep(actual_sleep_time)


app = flask.Flask(__name__)
app.json.ensure_ascii = False



def save_image_bytes(img_bytes, source="esp"):

    if source == "user":
        folder = "images_user_upload"
    elif source == "esp":
        folder = "images_esp_upload"
    else:
        folder = "images_unknown_sourse"

    time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    filename = time + ".jpg"

    save_folder = os.path.join(BASEPATH, folder)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    return filename, save_path

def upload_image(source:str="user"):
    if source == "user":
        folder = "images_user_upload"
    elif source == "esp":
        folder = "images_esp_upload"
    else:
        folder = "images_unknown_sourse"


    file = flask.request.files["image"]
    time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = time + "." + file.filename.split(".")[-1]
    save_path = os.path.join(BASEPATH,folder,filename)
    save_folder = os.path.join(BASEPATH,folder)
    os.makedirs(save_folder, exist_ok=True)

    file.save(save_path)

    return flask.jsonify({"filename": filename,
                          "time": time})


@app.route("/")
def api_home():

    routes = []

    for rule in sorted(
        app.url_map.iter_rules(),
        key=lambda x: str(x)
    ):

        if rule.endpoint != "static":
            routes.append(str(rule))

    return flask.Response(
        "\n".join(routes),
        mimetype="text/plain"
    )

@app.route("/test")
def api_test():
    return "別測試了好著呢"

@app.route("/simple_keyword_search/<keyword>")
def api_search_simple_keyword_search(keyword):
    results = search.simple_keyword_search(keyword)
    return flask.jsonify(results)

@app.route("/semantic_search/<keyword>")
def api_Semantic_search(keyword):
    embedding = search.text_to_embedding(keyword)
    results = search.embedding_similarity_search(embedding)
    return flask.jsonify(results)

@app.route("/image_search/<name>")
def api_Image_search(name):
    file = db.name_search(name)

    if file is None:
        return flask.jsonify({
            "error":"找不到圖片"
        }), 404
    
    path = file["img_path"]

    embedding = search.image_to_embedding(path)
    results = search.embedding_similarity_search(embedding)
    return flask.jsonify(results)

@app.route("/show_all")
def api_db_show_all():
    results = db.show_all()
    return flask.jsonify(results)

@app.route("/crop/<id>")
def api_db_view_crop(id):
    file = db.id_search(int(id))
    if file is None:
            return flask.jsonify({"error":"找不到ID"}), 404

    data = {"x1":file["x1"], "x2":file["x2"], "y1":file["y1"], "y2":file["y2"], "img_path":file["img_path"]}
    image = db.view_crop(**data)


    
    _, buffer = cv2.imencode(".jpg", image)

    return flask.Response(
        buffer.tobytes(),
        mimetype="image/jpeg"
    )

@app.route("/process/<imgname>")
def api_process_process_img(imgname):

    #防呆+增加彈性
    if imgname.isdigit():
        imgname = int(imgname)
        imgname = db.input_id_output_img_name(imgname)
        if imgname is None:
            return flask.jsonify({"error":"找不到ID"}), 404
        
    results = process.process_img(imgname)
    return flask.jsonify(results)

@app.route("/image/<filename_or_cropID>")
def get_user_image(filename_or_cropID):

    #防呆+增加彈性
    if filename_or_cropID.isdigit():
        cropID = int(filename_or_cropID)

        file = db.id_search(cropID)
        if file is None:
            return flask.jsonify({
                "error":"找不到ID"
            }), 404

    else:
        name = str(filename_or_cropID)
        file = db.name_search(name)

    if file:
        return flask.send_file(file["img_path"])
    return "找不到圖片，你有輸對ID或圖片名嗎？"

@app.route("/upload", methods=["POST"])
def api_upload_user():

    file = flask.request.files["image"]

    filename, path = save_image_bytes(img_bytes=file.read(), source="user")

    image_url = flask.url_for("get_user_image", filename_or_cropID=filename, _external=True)

    return flask.jsonify({
        "filename": filename,
        "path": path,
        "image_url": image_url
    })

@app.route("/esp_request")
def api_esp_request():
    if esp_status is None:
        return "esp未連線，請求失敗"
    
    img = esp_request.get_img()
    img_name, img_path = save_image_bytes(img_bytes=img, source="esp")
    

    folder_path = os.path.dirname(img_path)

    results = process.process_img(img_name=img_name, folder_path=folder_path)

    if "crops" in results:
        for crop in results["crops"]:
            crop.pop("CLIP_embedding", None)

    return flask.jsonify(results)




print("flask_server:完成前置任務")
if __name__ == "__main__":

    # threading.Thread(
    #     target=esp_threading,
    #     daemon=True
    # ).start()

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(
            target=esp_threading,
            kwargs={
                "ip": "192.168.4.1",
                "density": 11,
                "led": 0,
                "sleep_time": 5
                },
            daemon=True
        ).start()

    print("Now accessing DB:",end=" ")
    print(os.path.abspath("items.db"))
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )