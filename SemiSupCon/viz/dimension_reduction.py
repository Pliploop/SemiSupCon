# write two visualization functions for plotting learned embeddings of dimensions 512

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch.nn as nn
import seaborn as sns

import plotly.express as px
import plotly
import plotly.graph_objects as go





# def get_cosine_distances(representations):
#     # compute pairwise cosine similarity distance
#     distance = 1 - nn.functional.cosine_similarity(representations.unsqueeze(1),representations.unsqueeze(0),dim = 1)
#     return distance
    

# def plot_pca(representations, labels, title):
#     if isinstance(labels[0], list):
#         representations, labels = transform_for_multilabel(representations, labels, title)
        
#     # compute pairwise cosine similarity distance
    
    
        
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(representations)
#     plt.figure(figsize=(10,10))
#     scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=labels, cmap='tab20')
#     plt.legend(*scatter.legend_elements(), title="Classes")
#     plt.title(title)
#     plt.show()
    
    
def plot_tsne(representations, labels = None, title = None, cmap = 'tab20', perplexity = 40, n_iter = 300, metric = 'cosine', n_components = 2, *args, **kwargs):
    if labels is not None:
        if isinstance(labels[0], list):
            representations, labels = transform_for_multilabel(representations, labels, title)
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=perplexity, n_iter=n_iter, metric=metric)
    tsne_results = tsne.fit_transform(representations)
    fig,ax = plt.subplots(figsize=(10,10))
    # sns.scatterplot(x=for_viz['representations'][:,0], y=for_viz['representations'][:,1], hue=for_viz_pitch_classes, sizes = for_viz_octave, palette="deep")
    if n_components == 3:
        # interactive 3D plot with plotly
        # associate size with octave
        sizes = kwargs['size']
        # trace = px.scatter_3d(x=tsne_results[:,0], y=tsne_results[:,1], z=tsne_results[:,2], color = kwargs['hue'], size=sizes*3)
        layout = go.Layout(title=title, margin=dict(l=0, r=0, b=0, t=0), showlegend=True)
        hue = kwargs['hue']
        # convert the pandas series hue to class indices, where hue is a string
        hue = hue.astype('category').cat.codes
        hue = hue.to_numpy()/hue.max()
        trace = go.Figure(data=[go.Scatter3d(x=tsne_results[:,0], y=tsne_results[:,1], z=tsne_results[:,2], mode='markers', marker = dict(color=hue, size = sizes*2, opacity = 1, colorscale='rainbow', 
                                                                                                                                          line =dict(width=0, color='DarkSlateGrey')))], layout=layout)
        
        
        # fig = go.Figure(data=[trace], layout=layout)
        plotly.offline.iplot(trace)

    else:
        scatter = sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], ax = ax, *args, **kwargs)
        # plt.legend(loc='upper left', ncol=2)  
        # legend should be outside the plot to the left
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
        ax.axis('off')
        # no xticks and yticks and no x and y axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.title(title)
        
        return fig,ax
    
    
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
    
