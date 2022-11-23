from PIL import Image
import numpy as np

def RandomPatch(image,numpatches):
    imr = np.array(image.resize((224,224)))
    size = imr.shape[0]
    numpatch = int(np.sqrt(numpatches))
    patches = []
    for i in range(numpatch):
        for j in range(numpatch):
            patches.append(imr[i*int(size/numpatch):(i+1)*int(size/numpatch),j*int(size/numpatch):(j+1)*int(size/numpatch),:])
    randompatching = np.random.permutation(patches)
    newimage = np.zeros((size,size,3))
    k = 0
    for i in range(numpatch):
        for j in range(numpatch):
            newimage[i*int(size/numpatch):(i+1)*int(size/numpatch),j*int(size/numpatch):(j+1)*int(size/numpatch),:] = randompatching[k]
            k+=1
    newim = Image.fromarray(np.uint8(newimage))
    return newim