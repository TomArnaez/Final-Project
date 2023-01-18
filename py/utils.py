import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np
import torch
import urllib.request
from tqdm import tqdm

# Download method suggested by tqdm docs: https://github.com/tqdm/tqdm#hooks-and-callbacks
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def get_default_device():
    """ Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def download_mnist_c():
    urllib.request.urlretrieve("https://zenodo.org/record/3239543/files/mnist_c.zip", "./datasets/mnist_c.zip")
    
def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

def plot_training_loss(data, xlabel, ylabel, save_file):
    colors = ['red', 'blue', 'green', 'yellow', 'orange']
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    for d, color in zip(data, colors):
        plt.plot(d['train_counter'], d['train_losses'], color=color, label=d['label'])
    plt.savefig('./plots' + save_file, dpi=300, bbox_inches='tight')
    plt.legend(loc="upper right")
    print("Saving training loss plot at: " + "./plots/" + save_file)
    plt.show()

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print("utils")
