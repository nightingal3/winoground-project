from datasets import load_dataset
from matplotlib import pyplot as plt
import argparse
import pandas as pd

def plotImage(winoground,idx):
    ax1 = plt.subplot(1, 3, 1)
    ax1.title.set_text('image_0')
    plt.imshow(winoground[idx]["image_0"].convert("RGB"))

    ax2 = plt.subplot(1, 3, 2)
    ax2.title.set_text('image_1')
    plt.imshow(winoground[idx]["image_1"].convert("RGB"))

    plt.show()

def loadDataset(auth_token):
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    return winoground




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ac','--auth_token',type=str,required=True,help="Authorization token from huggingface")
    args = parser.parse_args()

    winoground=loadDataset(args.auth_token)
    for idx in range(320,400):
        print("caption_0:", winoground[idx]["caption_0"])
        print("caption_1:", winoground[idx]["caption_1"])
        plotImage(winoground,idx)
        #Next manually annotated the error category by analyzed the labels and images








