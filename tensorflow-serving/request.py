import os
from os import path
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gzip
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Download Test mnist fashion dataset
if not path.exists("data/t10k-images-idx3-ubyte.gz") or not path.exists("data/t10k-labels-idx1-ubyte.gz"):
  subprocess.run(["wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P data/"],shell=True)
  subprocess.run(["wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P data/"],shell=True)

# Reading Test Images
f = gzip.open('data/t10k-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 3
f.read(16)
buf = f.read(image_size * image_size * num_images)
img_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
img_data = img_data.reshape(num_images, image_size, image_size, 1)

# Reading Test labels
f = gzip.open('data/t10k-labels-idx1-ubyte.gz','r')
f.read(8)
test_labels = []
for i in range(0,3):   
    buf = f.read(1)
    test_labels.append(np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0])

def show(idx,title):
    image = np.asarray(img_data[idx]).squeeze()
    plt.imshow(image)
    plt.axis('off')
    plt.title('\n{}'.format(title), fontdict={'size': 9})
    plt.show()

with open('/path to the json file/predict.json') as f:
  data = json.load(f)
json_data = json.dumps(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/fashion_mnist:predict', data=json_data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)

for i in range(0,3):
    show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i]))
