import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from . import geometry as geom


COLOR_DEFAULT = mcolors.to_rgb("blue")
COLORS = [
    "lime",
    "red",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "brown",
    "pink",
]

def ind2color(ind):
    # Cycle through colors
    color_str = COLORS[ind % len(COLORS)]
    return mcolors.to_rgb(color_str)


class MatchPlotter:

    MARKER_SIZE = 5

    def __init__(self, im1=None, im2=None, matches=None, inlier_mask=None) -> None:
        self._im1 = None
        self._im2 = None
        self._x1 = None
        self._x2 = None
        self.masks = None

        if im1 is not None:
            self.add_image(im1, index=0)
        if im2 is not None:
            self.add_image(im2, index=1)
        if matches is not None:
            self.set_matches(matches, inlier_mask)


    def add_image(self, image_name, index=None):
        if index is None:
            if self._im1 is None:
                index = 0
            elif self._im2 is None:
                index = 1
            else:
                raise ValueError("Can not add image to plotter; all images are set. \
                                 If you want to replace an image, please specify index.")
        
        image = plt.imread(image_name)
        if index == 0:
            self._im1 = image
        elif index == 1:
            self._im2 = image
        else:
            raise ValueError("Invalid index")
    
    def add_images(self, image_names):
        for i, image_name in enumerate(image_names):
            self.add_image(image_name, index=i)

    def set_cameras(self, K1, K2):
        self.K1 = K1
        self.K2 = K2

    def set_matches(self, matches, inlier_mask=None, calibrated=False):
        x1, x2 = matches
        if calibrated:
            # Project back to image plane
            x1 = geom.uncalibrate_pts(x1, self.K1)
            x2 = geom.uncalibrate_pts(x2, self.K2)
        self._x1 = x1
        self._x2 = x2
        self.inlier_mask = inlier_mask

    def add_match(self, match):
        # Add a single match to existing matches
        new_x1, new_x2 = match
        if self._x1 is None:
            self._x1 = new_x1.copy().reshape(1, -1)
            self._x2 = new_x2.copy().reshape(1, -1)
        else:
            self._x1 = np.vstack([self._x1, new_x1])
            self._x2 = np.vstack([self._x2, new_x2])

    def add_matches(self, matches):
        # Add matches to existing matches
        raise NotImplementedError()

    def add_masks(self, masks):
        if self.masks is None:
            self.masks = masks.copy()
        else:
            self.masks += masks

    def draw_images(self):
        assert not self._im1 is None, "Image 1 not found"
        assert not self._im2 is None, "Image 2 not found"

        # Concatenate images
        h1, w1 = self._im1.shape[:2]
        h2, w2 = self._im2.shape[:2]
        img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=self._im1.dtype)
        img[:h1, :w1, :] = self._im1
        img[:h2, w1:, :] = self._im2

        # Plot image
        plt.figure(figsize=(20, 20))
        plt.imshow(img)


    def draw_inlier_matches(self):
        inlier_x1 = self._x1[self.inlier_mask]
        inlier_x2 = self._x2[self.inlier_mask]
        self.draw_matches(inlier_x1, inlier_x2)

    def draw_keypoints(self, x):
        if self.masks is None or len(self.masks) == 0:
            plt.scatter(x[:,0], x[:,1], s=3, color=COLOR_DEFAULT)
        else:
            for i, mask in enumerate(self.masks):
                color = ind2color(i)
                plt.scatter(x[mask,0], x[mask,1], s=self.MARKER_SIZE, color=color)
            unassigned_mask = ~sum(self.masks).astype(bool)
            plt.scatter(x[unassigned_mask,0], x[unassigned_mask,1], s=self.MARKER_SIZE, color=COLOR_DEFAULT)

    def draw_all_keypoints(self):
        if self._x1 is not None:
            # Keypoints in image 1
            self.draw_keypoints(self._x1)
        
        if self._x2 is not None and self._im1 is not None:
            # Keypoints in image 2 (offset by image 1)
            h, w = self._im1.shape[:2]
            x2 = self._x2.copy()
            x2[:,0] += w
            self.draw_keypoints(x2)

    def draw_matches(self, x1, x2, color=None):
        if x1 is None or x2 is None:
            return
        if self._im1 is None:
            return
        if color is None:
            color = COLOR_DEFAULT

        # Offset x2 by image 1
        x2 = x2.copy()
        _, w = self._im1.shape[:2]
        x2[:,0] += w

        for start, stop in zip(x1, x2):
            plt.plot([start[0], stop[0]], [start[1], stop[1]], color=color)


    def draw(self, title=None, show_matches=True, color=None):
        self.draw_images()

        if show_matches:
            self.draw_matches(self._x1, self._x2, color) 
        else:
            self.draw_all_keypoints()

        if title is None:
            title = f"{len(self._x1)} matches"
        plt.title(title)
        plt.show()


def make_heatmap(values, gt_values, bins=100, maxval=15):
    xmax = ymax = min(max(max(values), max(gt_values)), maxval)
    xrange = [0, xmax]
    yrange = [0, ymax]
    
    heatmap, xedges, yedges = np.histogram2d(values, gt_values, bins=bins, range=[xrange, yrange])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent, xedges, yedges

def draw_heatmap(ax, heatmap, extent, vmax=None, colorbar=True):
    PCM = ax.imshow(heatmap.T, extent=extent, origin='lower', vmax=vmax)
    if colorbar:
        plt.colorbar(PCM, ax=ax)
    ax.plot(extent[:2], extent[2:] ,'r--')
    ax.set_xlabel(r'Estimated $\sigma$')
    # ax.set_ylabel('GT rel. depth')
    ax.axis(extent)
    return PCM

def draw_logarithmic_heatmap(ax, heatmap, extent, xedges, yedges, vmax=None, colorbar=True):

    # Define colormap
    cmap = plt.colormaps["PuBu_r"]
    cmap.set_bad(cmap(0))
    
    X, Y = np.meshgrid(xedges, yedges)
    PCM = ax.pcolormesh(X, Y, heatmap.T, norm=mcolors.LogNorm(), cmap=cmap, rasterized=True)
    if colorbar:
        plt.colorbar(PCM, ax=ax)
    ax.plot(extent[:2], extent[2:] ,'r--')
    ax.set_xlabel(r'Estimated $\sigma$')
    # ax.set_ylabel('GT rel. depth')
    ax.axis(extent)
    return PCM

def plot_heatmaps(
        axs,
        gt_values,
        sift_values,
        pred_values,
        bins=100,
        maxval=2,
        vmax=0,
        logarithmic=False):

    heatmap_sift, extent_sift, xedges_sift, yedges_sift = make_heatmap(sift_values, gt_values, bins, maxval)
    heatmap_pred, extent_pred, xedges_pred, yedges_pred = make_heatmap(pred_values, gt_values, bins, maxval)
    
    if vmax <= 0:
        vmax = max(heatmap_sift.max(), heatmap_pred.max())
        print("Setting vmax", vmax)

    axs[0].set_title("SIFT")
    axs[1].set_title("RelScaleNet (Ours)")
    axs[0].set_ylabel(r'Ground Truth $\sigma$')
    axs[1].set_yticks([])

    if logarithmic:
        draw_logarithmic_heatmap(axs[0], heatmap_sift, extent_sift, xedges_sift, yedges_sift, vmax=vmax, colorbar=False)
        draw_logarithmic_heatmap(axs[1], heatmap_pred, extent_pred, xedges_pred, yedges_pred, vmax=vmax, colorbar=False)
    else:
        draw_heatmap(axs[0], heatmap_sift, extent_sift, vmax=vmax, colorbar=False)
        draw_heatmap(axs[1], heatmap_pred, extent_pred, vmax=vmax, colorbar=False)
