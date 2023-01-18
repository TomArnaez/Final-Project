import wget
import os

path = os.getcwd() + "/datasets"
print(f"Datasets will be downloaded to {path}")
mnist_c_url = "https://zenodo.org/record/3239543/files/mnist_c.zip"
print("Downloading MNIST-C dataset")
wget.download(mnist_c_url, out=path)

import zipfile
print("Extracting MNIST-C") 
with zipfile.ZipFile(path + "/mnist_c.zip", 'r') as zip_ref:
    zip_ref.extractall(path)

tiny_imagenet_c_url = "https://zenodo.org/record/2536630/files/Tiny-ImageNet-C.tar"
print("Downloading Tiny ImageNet-C dataset")
wget.download(tiny_imagenet_c_url, out=path)
print("Extracting Tiny ImageNet-C dataset")
with zipfile.ZipFile(path + "/tiny-imagenet-c.tar", 'r') as zip_ref:
    zip_ref.extractall(path)
