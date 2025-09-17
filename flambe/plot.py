import matplotlib.pyplot as plt
import pandas as pd 
#import seaborn as sns 
from matplotlib.colors import LogNorm

def hexbin_plot(data, x, y, ax=None, figsize=(5.5,5), cmap='cividis', log_hue=True, vmin=1, vmax=100, cbar_shrink=0.5, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if log_hue:
        kwargs['norm'] = LogNorm(vmin=vmin, vmax=vmax)
    
    hb = ax.hexbin(data[x], data[y], cmap=cmap, **kwargs)
    plt.colorbar(hb, shrink=cbar_shrink)
    
    return ax, hb