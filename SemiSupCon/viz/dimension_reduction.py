# write two visualization functions for plotting learned embeddings of dimensions 512

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE


def plot_pca(representations, labels, title):
    if isinstance(labels[0], list):
        representations, labels = transform_for_multilabel(representations, labels, title)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(representations)
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=labels, cmap='tab20')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.show()
    
    
def plot_tsne(representations, labels, title):
    if isinstance(labels[0], list):
        representations, labels = transform_for_multilabel(representations, labels, title)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(representations)
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='tab20')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.show()
    
    
#how would you deal with multilabel in this case? i.e one representation that has multiple labels
# repeat the representation as many times as there are labels and then plot the scatter plot
# rewrite the function to do this
# where labels is a list of lists

def transform_for_multilabel(representations, labels, title):
    new_labels = []
    for i in range(len(representations)):
        for j in range(len(labels[i])):
            if labels[i][j] == 1:
                if j == 0:
                    new_representations = representations[i]
                    new_labels.append(labels[i][j])
                else:
                    new_representations = torch.cat([new_representations,representations[i]])
                    new_labels.append(labels[i][j])
                    
    return new_representations, new_labels
    
