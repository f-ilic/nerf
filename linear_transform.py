import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange 
from matplotlib.widgets import Slider, Button

samples = 100
xx, yy = np.meshgrid(np.linspace(-1,1,samples),np.linspace(-1,1,samples))
grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
r = (yy.flatten()*(1/2)) + 1/2
g = (xx.flatten()*(1/2)) + 1/2
b = np.sqrt(xx**2 + yy**2).flatten()
b = b/max(b)

colors = torch.from_numpy(np.stack([b,g,r])).unbind(1)
colors = [ [x.item() for x in c]  for c in colors]

I = np.array([
              [1, 0], 
              [0, 1]  
              ])

A = np.array([
              [-4, -0.5],
              [0, 2] 
              ])

print(I)
print(A)

fig, ax = plt.subplots(figsize=(4,4))
ax.set_aspect(1)
plt.grid(which='major')

# plt.subplots_adjust(bottom=0.45)

ax_interpolation = plt.axes([0.2, 0.9, 0.65, 0.03])
interpolation_slider = Slider(
    ax=ax_interpolation,
    label='interpol.',
    valmin=0,
    valmax=1,
    valinit=0,
)


def update(val):
    D = I+((A-I)*val)
    out = D @ grid.T
    im.set_offsets(out.T)


interpolation_slider.on_changed(update)

out = I @ grid.T
im = ax.scatter(out[0,:], out[1,:], c=colors, alpha=1, s=0.4)
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)

plt.show(block=True)
