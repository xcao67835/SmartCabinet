#我的產能真的超高的
#在vscode看.db:裝"SQLite Viewer"外掛(還是叫做插件我忘了)
import sqlite3
import json
import os
import cv2
import flask
#創database(items.db)
def create_table():
    connect = sqlite3.connect("items.db")
    cursor = connect.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
                   
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    img_name TEXT,
    img_path TEXT,           

    YOLO_confidence REAL,
    x1 INTEGER,
    y1 INTEGER,
    x2 INTEGER,
    y2 INTEGER,

    clip_describe TEXT,
    blip_describe TEXT,

    embedding TEXT
                   
    )
    """)

    connect.commit()
    connect.close()

#清除items.db 不是把items.db刪掉而是把資料清掉 欄位名稱還留著
def clear_db():

    connect = sqlite3.connect("items.db")
    cursor = connect.cursor()

    try:
        cursor.execute("""
        DELETE FROM items
        """)
    except Exception as e:
        print('db.clear_db錯誤：資料庫沒刪成功')
        print(e)

    connect.commit()
    connect.close()


def view_crop(img_path:str, x1:int, x2:int, y1:int, y2:int, show_original:bool=False, view_in_window:bool=False):

    readimg = cv2.imread(img_path)
    if readimg is None:
        raise FileNotFoundError(img_path)
    image = readimg.copy()

    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if show_original:

        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            3
        )
        result = image

    else:
        result = image[y1:y2, x1:x2]

    if view_in_window:
        cv2.imshow("crop", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result

#丟到創好的db
def save_to_db(data:dict):
    #print("db.save_to_db 執行中")
    connect = sqlite3.connect("items.db")
    cursor = connect.cursor()
    for crop in data["crops"]:
        cursor.execute("""
        INSERT INTO items
        (img_name, img_path, YOLO_confidence, x1, y1, x2, y2, clip_describe, blip_describe, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

        data["img_name"],
        data["img_path"],
        crop["YOLO_confidence"],
        crop["crop_position"]["x1"],
        crop["crop_position"]["y1"],
        crop["crop_position"]["x2"],
        crop["crop_position"]["y2"],
        crop["CLIP_describe"],
        crop["BLIP_describe"],
        json.dumps(crop["CLIP_embedding"])

        ))

    connect.commit()
    connect.close()
    #print("db.save_to_db 執行完成")
def id_search(id:int):
    connect = sqlite3.connect("items.db")
    connect.row_factory = sqlite3.Row
    cursor = connect.cursor()

    cursor.execute("""
    SELECT *
    FROM items
    WHERE id = ?
    """, (id,))

    row = cursor.fetchone()
    connect.close()

    if row is None:
        print("沒是這ID的圖片")
        return None
    
    result = dict(row)
    return result

def name_search(img_name:str):
    connect = sqlite3.connect("items.db")
    connect.row_factory = sqlite3.Row
    cursor = connect.cursor()

    cursor.execute("""
    SELECT *
    FROM items
    WHERE img_name = ?
    """, (img_name,))

    row = cursor.fetchone()
    connect.close()

    if row is None:
        print("沒是這名字的圖片")
        return None
    
    result = dict(row)
    return result

#好像挺無用的但我做了
def input_id_output_img_name(id:int):
    file = id_search(id)
    if file is None:
        return None

    img_name = file["img_name"]
    return img_name

#print全部db(沒有embedding) memory占用大
def show_all():
    """
    顯示資料庫內除了embedding外全部資料
    資料庫:items.db
    """

    connect = sqlite3.connect("items.db")
    connect.row_factory = sqlite3.Row
    cursor = connect.cursor()

    cursor.execute("""
    SELECT 
    *
    FROM items
    """)

    # embedding太大 我想把所有資料除了embedding全部弄出來
    # 但fetchall還是會吃到一下下的全部 包含embedding
    # 所以用fetchone一條一條用
    results = []
    while True:
        row = cursor.fetchone()
        if row is None:
            break

        row = dict(row)
        row.pop("embedding", None)
        results.append(row)

    connect.close()

    return results

#刪掉原本的db然後把 JSON_results_folder 裡面的.json都抓出來重建db
def rebuild_db():

    clear_db()

    for file_name in os.listdir("JSON_results_folder"):

        if file_name.endswith(".json"):
            with open(

                f"JSON_results_folder/{file_name}",
                "r",
                encoding="utf-8"

            ) as f:

                data = json.load(f)
            save_to_db(data)

print("db.py accessing DB:", end = " ")
print(os.path.abspath("items.db"))



if __name__ == "__main__":
    create_table()
    rebuild_db()
