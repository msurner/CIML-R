import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import viridis, ScalarMappable


def weight_copy(src_model, dst_model):
    """ 
    Copy parameter (weights and bias) values from one model to another.
    Params:
    - src_model (torch.nn.Module): Source of weight values
    - dst_model (torch.nn.Module): Destination of weight values
    """
    for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
        dst_param.data.copy_(src_param.data)

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def pprint(d, indent=2):
    """
    Pretty print a directory in JSON syntax.
    """
    print(json.dumps(d, cls=TensorEncoder, indent=indent))

def plot_hist(losses, accs, intervened_accs=None, epochs=None, fig=None, ax1=None, ax2=None, model_name="", plot_legend=False):
    if len(losses) != len(accs):
        raise ValueError("length of losses and accs must be equal.")

    if epochs is None:
        epochs = range(1, len(losses)+1)

    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots()

    if ax2 is None:
        ax2 = ax1.twinx()

    plt.grid(True)
    ax1.plot(epochs, losses, color=viridis(0.), linestyle="solid", label=r"$\mathbf{V}_o$ loss")
    ax2.plot(epochs, accs, color=viridis(0.35), linestyle="--", label=r"$\mathbf{V}_o$ accuracy")
    if intervened_accs is not None:
        ax2.plot(epochs, intervened_accs, color=viridis(.7), linestyle=":", label=r"$\mathbf{V}_i$ accuracy")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    ax1.set_title(f"Training Performance {model_name}")

    if plot_legend:
        legend = fig.legend(
            bbox_to_anchor=(.5, -0.07),
            loc="lower center",
            bbox_transform=fig.transFigure,
            ncol=3
        )

    return fig, legend

def df_append(df, row):
    """ 
    Appends row to a dataframe.
    Args:
    - df: pd.DataFrame
    - row: dict
    Returns:
    - pd.DataFrame
    """
    df_row = pd.DataFrame(data=[row])
    return pd.concat([df, df_row])

def set_seeds(seed):
    """
    Set a seed all potential sources of random numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def plot_time_dist(values, title, ylabel, nr_quantiles=10, vmin=0., vmax=1.):
    """
    Plot time distribution over training epochs.
    """

    N = nr_quantiles
    cmap = viridis
    quantiles = [i/float(N) for i in range(N+1)]
    fig, ax = plt.subplots()
    ax.grid(True)
    x = np.arange(1, values.shape[1]+1)
    for q in quantiles:
        quantile = np.quantile(values, q, axis=0)
        ax.plot(x, quantile, color=cmap(q), label=f"{q}")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    fig.colorbar(sm, ax=ax, ticks=quantiles,
                 boundaries=np.arange(-0.05,1.1,.1));
    return fig