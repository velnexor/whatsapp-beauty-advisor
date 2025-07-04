import os
import requests

# Example image URLs for each class (replace with your own or real dataset links)
SAMPLES = {
    'train/oval': [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    ],
    'train/round': [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    ],
    'train/square': [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    ],
    'validation/oval': [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    ],
    'validation/round': [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    ],
    'validation/square': [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    ]
}

for folder, urls in SAMPLES.items():
    os.makedirs(f'dataset/{folder}', exist_ok=True)
    for i, url in enumerate(urls, 1):
        img_path = f'dataset/{folder}/sample_{i}.jpg'
        r = requests.get(url)
        with open(img_path, 'wb') as f:
            f.write(r.content)
        print(f'Downloaded {img_path}')
