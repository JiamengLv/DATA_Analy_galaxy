import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import tqdm


def generate_urls():

    base_url = "http://jsoc.stanford.edu/data/aia/synoptic"
    start_date = datetime(2010, 5, 13)
    end_date = datetime(2020, 5, 14)  
    wavelengths = ["0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600", "1700", "4500"]

    urls = []
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y/%m/%d")
        date_file_str = current_date.strftime("%Y%m%d")
        for hour in range(0, 24):  
            time_str = f"H{hour:02}00"
            for minute in range(0, 60, 12):  
                minute_str = f"{minute:02}"
                for wavelength in wavelengths:
                    url = f"{base_url}/{date_str}/{time_str}/AIA{date_file_str}_00{minute_str}_{wavelength}.fits"
                    urls.append(url)
        current_date += timedelta(days=1)

    return urls

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {url}")
    except requests.exceptions.RequestException as e:
        with open("error_urls.txt", "a") as f:
            f.write(f"{url}\n")
    



def main():
    urls = generate_urls()
    save_dir = "downloads"
    os.makedirs(save_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=2) as executor:  
        futures = {}
        for url in tqdm.tqdm(urls):
            parts = url.split('/')
            date_dir = parts[-5]+parts[-4]+parts[-3]  
            time_dir = parts[-2] 
            file_name = parts[-1]  

            save_path = os.path.join(save_dir, date_dir, time_dir, file_name)
            
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                  os.makedirs(dir_path)
            futures[executor.submit(download_file, url, save_path)] = url

        for future in as_completed(futures):
            url = futures[future]
            try:
                future.result()  
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')

if __name__ == "__main__":
    main()