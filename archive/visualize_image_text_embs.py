import numpy as np
import pdb
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("./data/clip_embs/image.npy", "rb") as f:
        image_embs = np.load(f)
        image_embs_flat = image_embs.reshape((image_embs.shape[0] * image_embs.shape[1], 512))
    with open("./data/clip_embs/text.npy", "rb") as f:
        text_embs = np.load(f)
        text_embs_flat = text_embs.reshape((text_embs.shape[0] * text_embs.shape[1], 512))

    tsne = TSNE(n_components=2, random_state=42)
    labels = ["image"] * len(image_embs_flat) + ["text"] * len(text_embs_flat)
    all_embs = np.concatenate([image_embs_flat, text_embs_flat])
    embs_transformed = tsne.fit_transform(all_embs)

    plt.scatter(embs_transformed[:len(image_embs_flat), 0], embs_transformed[:len(image_embs_flat), 1], color="orange", label="Image", s=1)
    plt.scatter(embs_transformed[len(image_embs_flat):, 0], embs_transformed[len(image_embs_flat):, 1], color="blue", label="Text", s=1)
    plt.legend()
    plt.savefig("./image_and_text_tsne.png")
    plt.savefig("./image_and_text_tsne.eps")

    plt.gcf().clear()

    first_20_images = embs_transformed[:20]
    first_20_texts = embs_transformed[len(image_embs_flat):len(image_embs_flat) + 20]

    colours = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i,(img, txt) in enumerate(zip(first_20_images, first_20_texts)):
        if i % 2 == 0:
            c = next(colours)
        plt.plot(img[0], img[1], color=c, marker="o")
        plt.annotate(str((i % 2) + 1), (img[0], img[1]))
        plt.plot(txt[0], txt[1], color=c, marker="s")
        plt.annotate(str((i % 2) + 1), (txt[0], txt[1]))
    
    img_mark = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, label='Image')
    txt_mark = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                            markersize=10, label='Text')

    plt.legend(handles=[img_mark, txt_mark])

    plt.savefig("image_text_pairs.png")
    plt.savefig("image_text_pairs.eps")



