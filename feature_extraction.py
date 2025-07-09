import cv2
import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage.io import imread

def extract_features(img_path):
    img = imread(img_path)
    img_gray = rgb2gray(img)
    features = []

    # Color features
    features.extend(np.mean(img, axis=(0, 1)))
    features.extend(np.std(img, axis=(0, 1)))

    # LBP
    lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= hist.sum()
    features.extend(hist)

    # Haralick
    glcm = greycomatrix((img_gray * 255).astype('uint8'), [1], [0], 256, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']:
        features.append(greycoprops(glcm, prop)[0, 0])

    return features
