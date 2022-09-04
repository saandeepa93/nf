from sys import exit as e
from icecream import ic

import torch 
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, log
import os


def plot_3d(x, prob, k, target):
  target = [str(i) for i in target.detach().numpy()]
  df = pd.DataFrame(x, columns=["x0", "x1"])
  df_2 = pd.DataFrame(prob, columns=["pdf"])
  df_color = pd.DataFrame(target, columns=["color"])
  df = df.join(df_2)
  df = df.join(df_color)
  fig = px.scatter_3d(df, x='x0', y='x1', z='pdf', color='color', title=f"PDF")
  fig.update_traces(marker=dict(size=6))
  fig.write_html(f"./plots/html/{k}.html")


def plot(arr, labels, ep):
  if labels is None:
    plt.scatter(arr[:, 0], arr[:, 1])
  else:
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i in range(len(arr)):
      plt.scatter(arr[i, 0], arr[i, 1], color = colors[labels[i]])
  plt.savefig(f"./plots/{ep}")
  plt.close()