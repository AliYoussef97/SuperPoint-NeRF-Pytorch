import requests
from tqdm import tqdm
import time
import sys
from requests.exceptions import RequestException, ConnectionError

url =   "http://images.cocodataset.org/zips/train2014.zip"  #"http://images.cocodataset.org/zips/val2014.zip" #"http://images.cocodataset.org/zips/test2014.zip"
filename = "train2014.zip" #"val2014.zip" #"test2014.zip"

total_retries = 5
retry_delay = 5  # seconds

response = None

for _ in range(total_retries):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        break
    except (RequestException, ConnectionError) as e:
        print(f"Error occurred: {e}. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
else:
    print(f"Failed to establish a connection after {total_retries} retries. Aborting.")
    sys.exit(1)

total_size = int(response.headers.get('content-length', 0))

progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

with open(filename, 'wb') as file:
    for data in response.iter_content(1024):
        progress_bar.update(len(data))
        file.write(data)

progress_bar.close()
print("Download complete.")