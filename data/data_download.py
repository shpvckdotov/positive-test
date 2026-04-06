import os
import requests
import argparse

def download_file(url, destination):
    """Скачивает файл по URL и сохраняет в destination."""
    print(f"Downloading {url}...")
    
    # Создаем директорию, если её нет
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"Saved to {destination}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download log datasets")
    parser.add_argument("--output-dir", default="data/sample_logs", help="Directory to save logs")
    args = parser.parse_args()

    # Ссылки на Raw файлы с GitHub (примеры, проверь актуальные ссылки на loghub)
    datasets = {
        "linux.log": "https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log",
        "windows.log": "https://raw.githubusercontent.com/logpai/loghub/master/Windows/Windows_2k.log",
        "hdfs.log": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log" 
    }

    for filename, url in datasets.items():
        dest_path = os.path.join(args.output_dir, filename)
        if not os.path.exists(dest_path):
            try:
                download_file(url, dest_path)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping.")