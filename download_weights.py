import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def main():
    # 目标目录
    save_dir = os.path.join('experiments', 'pretrained_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # GFPGAN v1.3
    model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    model_path = os.path.join(save_dir, 'GFPGANv1.3.pth')
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
    else:
        print(f"Downloading GFPGAN v1.3 model to {model_path}...")
        download_file(model_url, model_path)
        print("Download complete.")

if __name__ == "__main__":
    main()

