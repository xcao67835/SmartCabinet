import requests

esp_ip = "192.168.4.1"





"""
~&val=
0  -> QQVGA 160x120
3  -> HQVGA 240x176
5  -> QVGA 320x240
8  -> VGA 640x480
9  -> SVGA 800x600
10 -> XGA 1024x768
11 -> HD 1280x720
13 -> UXGA 1600x1200
"""
def set_frame_size(val, ip=None):

    if ip is None:
        ip = esp_ip

    requests.get(f"http://{ip}/control?var=framesize&val={val}")

def set_led_intensity(val, ip=None):

    if ip is None:
        ip = esp_ip

    requests.get(f"http://{ip}/control?var=led_intensity&val={val}")

def request_esp_status(ip=None):
    if ip is None:
        ip = esp_ip

    try:

        response = requests.get(
            f"http://{ip}/status",
            timeout=2
        )

        return response

    except Exception as e:
        print(e)
        return None

def get_img(ip=None):

    if ip is None:
        ip = esp_ip

    img = requests.get(f"http://{ip}/capture", timeout=10)
    img.raise_for_status()

    return img.content

if __name__ == "__main__":
    img = get_img()
    with open("test.jpg","wb") as f:
        f.write(img)