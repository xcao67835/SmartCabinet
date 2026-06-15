import sqlite3
import json
from PIL import Image
import cv2
from models_load import clip_processor, clip_model



#這是...向量內積攻勢!
def cosine_similarity(values1:list, values2:list):
    return sum(value1*value2 for value1,value2 in zip(values1,values2))

#我根本沒看懂這在幹嘛
def text_to_embedding(text:str):

    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    text_features = clip_model.get_text_features(**inputs)
    text_features = text_features.pooler_output
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    return text_features[0].tolist()

def image_to_embedding(img_path):
    image = cv2.imread(img_path) 
    if image is None:
        raise FileNotFoundError(img_path)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    inputs = clip_processor(
        images=image_pil,
        return_tensors="pt",
        padding=True    
    )

    # print("==============")
    # print(type(clip_model))
    # print(type(inputs))
    # print(inputs.keys())

    # 
    
    # print(type(image_features))
    # print(image_features)
    # print(clip_model.get_image_features)
    # print("==============")
    image_features = clip_model.get_image_features(**inputs)
    image_features = image_features.pooler_output
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    return image_features[0].tolist()


#簡單關鍵字搜尋
def simple_keyword_search(keyword, field="blip_describe"):

    connect = sqlite3.connect("items.db")
    connect.row_factory = sqlite3.Row
    cursor = connect.cursor()


    existing_field = []
    cursor.execute("""
    PRAGMA table_info(items)
    """)
    for i in cursor.fetchall():
        existing_field.append(i[1])

    if field not in existing_field:
        print("search.simple_keyword_search錯誤：沒這欄位")
        connect.close()
        return None


    sql = f"""
    SELECT *
    FROM items
    WHERE {field} LIKE ?
    """

    cursor.execute(sql, (f"%{keyword}%",))
    rows = cursor.fetchall()

    connect.close()

    print(f"db.search {keyword} in {field} return:",end='')

    results = []
    for row in rows:
        row = dict(row)
        row.pop("embedding", None)
        results.append(row)

    if len(results) == 0:
        print("找不到資料")
    return results



#以圖搜圖/以文搜圖  用向量比較相似度
def embedding_similarity_search(key_embedding_value:list, db:str="items.db"):
    """
    使用CLIP embedding計算相似度

    input:
        key_embedding_value : list[float]

    output:
        list[dict]

    return:
        依照score由大到小排序
    """

    connect = sqlite3.connect(db)
    connect.row_factory = sqlite3.Row
    cursor = connect.cursor()

    results = []

    cursor.execute("""
    SELECT *
    FROM items
    """)

    for row in cursor:
        row = dict(row)
        db_embedding_value = json.loads(row["embedding"])

        score = cosine_similarity(key_embedding_value, db_embedding_value)
        row["score"] = score

        row.pop("embedding", None)

        results.append(row)

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


if __name__ == "__main__":
    text_embedding = text_to_embedding("phone")
    results = embedding_similarity_search(text_embedding)

    for result in results[:5]:
        print(
            result["score"],
            result["img_name"],
            result["clip_describe"],
            result["blip_describe"]
        )