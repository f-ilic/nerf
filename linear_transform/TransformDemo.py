import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange 
from scipy.linalg import polar
from matplotlib.widgets import Slider, Button
np.set_printoptions(precision=1)

class LinearTransform():
    def __init__(self, A, data, colors=None) -> None:
        self.colors = colors
        self.data = data
        self.A = A
        self.I = np.array([
                            [1, 0], 
                            [0, 1]  
                          ])

        self.ihat = self.I[:,0]*0.1
        self.jhat = self.I[:,1]*0.1

        self.fig, self.ax = plt.subplots(figsize=(4,4))
        self.ax.set_aspect(1)

        plt.subplots_adjust(top=0.65)

        major_ticks = np.arange(-1, 1, 0.20)
        minor_ticks = np.arange(-1, 1, 0.05)

        self.ax.set_xticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
        self.ax.set_yticks(minor_ticks, minor=True)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        self.ax.grid(which='both')

        self.ax.grid(which='minor', alpha=0.2)
        self.ax.grid(which='major', alpha=0.5)


        self.ax_interpolation = plt.axes([0.2, 0.9, 0.65, 0.03])
        self.interpolation_slider = Slider(ax=self.ax_interpolation, label='interpol.', valmin=0, valmax=1, valinit=0)

        self.interpolation_slider.on_changed(self.update)

        out = self.I @ self.data.T

        self.ax.plot([0, self.ihat[0]], [0, self.ihat[1]], '-r', alpha=0.2)
        self.ax.plot([0, self.jhat[0]], [0, self.jhat[1]], '-b', alpha=0.2)
        self.im_i, = self.ax.plot([0, self.ihat[0]], [0, self.ihat[1]], '-r')
        self.im_j, = self.ax.plot([0, self.jhat[0]], [0, self.jhat[1]], '-b')

        self.im = self.ax.scatter(out[0,:], out[1,:], c=self.colors, alpha=1, s=40)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(-1, 1)

        self.ax_text = self.ax.text(-1.1, 1.4, str(self.I))
        # plt.show(block=True)
        self.fig.show()


    def update(self, val):
        D = self.I+((self.A-self.I)*val)
        out = D @ self.data.T
        self.im.set_offsets(out.T)

        self.ihat = D[:,0]*0.1
        self.jhat = D[:,1]*0.1
        self.im_i.set_data([0, self.ihat[0]], [0, self.ihat[1]])
        self.im_j.set_data([0, self.jhat[0]], [0, self.jhat[1]])

        self.ax_text.set_text(str(D))




class LinearTransformSeperated():
    def __init__(self, A, data, colors=None) -> None:
        self.colors = colors
        self.data = data
        self.A = A
        self.I = np.array([
                            [1, 0], 
                            [0, 1]  
                          ])

        self.ihat = self.I[:,0]*0.1
        self.jhat = self.I[:,1]*0.1

        self.fig, self.ax = plt.subplots(figsize=(4,4))
        self.ax.set_aspect(1)

        plt.subplots_adjust(top=0.65)

        major_ticks = np.arange(-1, 1, 0.20)
        minor_ticks = np.arange(-1, 1, 0.05)

        self.ax.set_xticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
        self.ax.set_yticks(minor_ticks, minor=True)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])


        self.ax.grid(which='both')

        self.ax.grid(which='minor', alpha=0.2)
        self.ax.grid(which='major', alpha=0.5)

        self.ax_R = plt.axes([0.2, 0.9, 0.65, 0.03])
        self.R_slider = Slider(ax=self.ax_R,label='Rotat',valmin=0,valmax=1,valinit=0)

        self.ax_S = plt.axes([0.2, 0.85, 0.65, 0.03])
        self.S_slider = Slider(ax=self.ax_S,label='Scale',valmin=0,valmax=1,valinit=0)

        self.ax_total = plt.axes([0.2, 0.70, 0.65, 0.03])
        self.total_slider = Slider(ax=self.ax_total,label='Total',valmin=0,valmax=1,valinit=0)

        self.total_slider.on_changed(self.update)
        self.S_slider.on_changed(self.update_indep)
        self.R_slider.on_changed(self.update_indep)

        out = self.I @ self.data.T

        self.ax.plot([0, self.ihat[0]], [0, self.ihat[1]], '-r', alpha=0.2)
        self.ax.plot([0, self.jhat[0]], [0, self.jhat[1]], '-b', alpha=0.2)
        self.im_i, = self.ax.plot([0, self.ihat[0]], [0, self.ihat[1]], '-r')
        self.im_j, = self.ax.plot([0, self.jhat[0]], [0, self.jhat[1]], '-b')

        self.im = self.ax.scatter(out[0,:], out[1,:], c=self.colors, alpha=1, s=40)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(-1, 1)

        self.ax_text = self.ax.text(-1.1, 1.4, str(self.I))
        self.fig.show()


    def update_indep(self, val):
        
        rotation, scale = polar(A)
        r_val = self.R_slider.val
        s_val = self.S_slider.val

        R = (self.I+(rotation-self.I)*r_val)
        S = (self.I+ (scale-self.I)*s_val)
        D = R @ S
        out = D @ self.data.T
        self.im.set_offsets(out.T)

        self.ihat = D[:,0]*0.1
        self.jhat = D[:,1]*0.1
        self.im_i.set_data([0, self.ihat[0]], [0, self.ihat[1]])
        self.im_j.set_data([0, self.jhat[0]], [0, self.jhat[1]])

        self.ax_text.set_text(str(D))

    def update(self, val):
        D = self.I+((self.A-self.I)*val)
        out = D @ self.data.T
        self.im.set_offsets(out.T)

        self.ihat = D[:,0]*0.1
        self.jhat = D[:,1]*0.1
        self.im_i.set_data([0, self.ihat[0]], [0, self.ihat[1]])
        self.im_j.set_data([0, self.jhat[0]], [0, self.jhat[1]])
        self.R_slider.set_val(val)
        self.S_slider.set_val(val)
        self.ax_text.set_text(str(D))


if __name__ == "__main__":
    samples = 100
    xx, yy = np.meshgrid(np.linspace(-1,1,samples, endpoint=False),np.linspace(-1,1,samples, endpoint=False))
    grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
    r = (yy.flatten()*(1/2)) + 1/2
    g = (xx.flatten()*(1/2)) + 1/2
    b = np.sqrt(xx**2 + yy**2 - yy**4 + xx**2).flatten()
    b = b/max(b)

    colors = torch.from_numpy(np.stack([b,g,r])).unbind(1)
    colors = [ [x.item() for x in c]  for c in colors]

    A = np.array([
                [0, 7],
                [-7, 0] 
                ])

    l1 = LinearTransform(A, grid, colors)
    l2 = LinearTransformSeperated(A, grid, colors)
    plt.show()