import gdown
import os

FOLDER_ID = "1JVTWilaSLsCt2ktT13RTafxclLixvr6-"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

folder_url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
gdown.download_folder(folder_url, output=OUT_DIR, quiet=False, use_cookies=False)
