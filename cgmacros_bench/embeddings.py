"""MobileNetV2 (ImageNet) embeddings of the meal photographs, cached on disk. Requires tensorflow and the images
that ship with CGMacros. The images are not redistributed here."""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import N_PCA, SEED
from .utils import log


def get_embeddings(uniq, cache_dir):
    cache = os.path.join(cache_dir, "mobilenet_emb.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == uniq:
            return z["emb"]
    log("Computing MobileNetV2 embeddings for", len(uniq), "images ...")
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.utils import img_to_array, load_img
    net = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    embs = []
    for i in range(0, len(uniq), 32):
        b = []
        for p in uniq[i:i + 32]:
            try:
                b.append(preprocess_input(img_to_array(load_img(p, target_size=(224, 224)))))
            except Exception:                                   # unreadable image: zero input
                b.append(np.zeros((224, 224, 3), np.float32))
        embs.append(net.predict(np.stack(b), verbose=0))
    emb = np.concatenate(embs).astype(np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, emb=emb, paths=np.array(uniq, dtype=object))
    return emb


def image_features(emb, img_idx, train_rows):
    """PCA fitted on the meal images of the training-fold participants only; returns (origins x N_PCA) features
    (zeros for origins without a previous meal photograph) and their column names."""
    tr_imgs = np.unique(img_idx[train_rows])
    tr_imgs = tr_imgs[tr_imgs >= 0]
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(emb[tr_imgs])
    Z = pca.transform(emb).astype(np.float32)
    X = np.where((img_idx >= 0)[:, None], Z[np.clip(img_idx, 0, None)], 0.0)
    return X.astype(np.float32), [f"img_pc{i}" for i in range(N_PCA)]
