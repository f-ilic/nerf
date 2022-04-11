import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange 
from matplotlib.widgets import Slider, Button
np.set_printoptions(precision=2)
samples = 100
xx, yy = np.meshgrid(np.linspace(-1,1,samples, endpoint=False),np.linspace(-1,1,samples, endpoint=False))
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

# A = np.array([
#               [0, 1],
#               [1, 0] 
#               ])

A = np.array([
              [0, 7],
              [-7, 0] 
              ])

print(I)
print(A)

fig, ax = plt.subplots(figsize=(4,4))
ax.set_aspect(1)
plt.subplots_adjust(top=0.65)

major_ticks = np.arange(-1, 1, 0.20)
minor_ticks = np.arange(-1, 1, 0.05)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

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
    global ihat
    global jhat
    D = I+((A-I)*val)
    out = D @ grid.T
    im.set_offsets(out.T)

    ihat = D[:,0]*0.1
    jhat = D[:,1]*0.1
    im_i.set_data([0, ihat[0]], [0, ihat[1]])
    im_j.set_data([0, jhat[0]], [0, jhat[1]])
    eigenvalues, eigenvectors = np.linalg.eig(D)
    print(eigenvalues* eigenvectors)
    ax_text.set_text(str(D))
interpolation_slider.on_changed(update)

out = I @ grid.T
ihat = I[:,0]*0.1
jhat = I[:,1]*0.1
ax.plot([0, ihat[0]], [0, ihat[1]], '-r', alpha=0.2)
ax.plot([0, jhat[0]], [0, jhat[1]], '-b', alpha=0.2)
im_i, = ax.plot([0, ihat[0]], [0, ihat[1]], '-r')
im_j, = ax.plot([0, jhat[0]], [0, jhat[1]], '-b')

im = ax.scatter(out[0,:], out[1,:], c=colors, alpha=1, s=4.4)
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)

ax_text = ax.text(-0.9, 0.8, str(I))

plt.show(block=True)
