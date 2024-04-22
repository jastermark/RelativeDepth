import torch
import numpy as np
import pickle
import cv2
import os
import h5py
import time
import torchvision

from .relscalenet import RelScaleNet


def extract_patch(tensor, bbox):
    """ Extracts a path from tensor with a given box.

    Args:
        tensor ((N, C, H, W)-float32 torch.Tensor):
        bbox (int32): [xmin, xmax, ymin, max]
    """

    assert torch.is_tensor(tensor)
    _, _, height, width = tensor.shape

    xmin, xmax, ymin, ymax = bbox
    
    pad_left = -torch.minimum(torch.tensor(0, dtype=xmin.dtype), xmin)
    pad_top = -torch.minimum(torch.tensor(0, dtype=ymin.dtype), ymin)
    pad_right = -torch.minimum(torch.tensor(0, dtype=xmax.dtype), width-xmax)
    pad_bottom = -torch.minimum(torch.tensor(0, dtype=ymax.dtype), height-ymax)
    padding = [pad_left, pad_top, pad_right, pad_bottom]
    padded_tensor = torchvision.transforms.Pad(padding)(tensor)

    xmin = xmin + pad_left
    xmax = xmax + pad_left
    ymin = ymin + pad_top
    ymax = ymax + pad_top

    return padded_tensor[:, :, ymin:ymax, xmin:xmax]


def find_patch(center_point, size):
    """ Finds a patch of a given size around a center point.

    Args:
        center_point (_type_): _description_
        size (_type_): _description_
    """
    if not torch.is_tensor(center_point):
        center_point = torch.tensor(center_point)
    if not torch.is_tensor(size):
        size = torch.tensor(size)

    # Convert to float64 to avoid rounding errors
    center_point = center_point.to(torch.float64)
    size = size.to(torch.float64)

    xc, yc = center_point + 0.5
    dx = dy = size/2  # TODO: Support non-square patches

    int_type = torch.int32
    xmin = torch.floor(xc - dx).to(int_type)
    xmax = torch.floor(xc + dx).to(int_type)
    ymin = torch.floor(yc - dy).to(int_type)
    ymax = torch.floor(yc + dy).to(int_type)

    assert xmax - xmin == size, "Patch width is not equal to size"
    assert ymax - ymin == size, "Patch height is not equal to size"
    return xmin, xmax, ymin, ymax

def extract_patches(im, x, size=64):
    patches = []
    for center in x:
        bbox = find_patch(center, size)
        patches.append(extract_patch(im, bbox))

    return patches
    

def load_image(im_path, resize=False, max_dim=640):
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    if resize:
        scale = max_dim / np.max(im.shape[0:2])
        new_dim = (int(im.shape[0]*scale), int(im.shape[1]*scale))
        im = cv2.resize(im, new_dim)
    else:
        scale = 1

    # (1, C, H, W) RGB image
    im = im.transpose([2, 0, 1])
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    return im, scale

def write_recursive_hdf5(f, data):
    for key, value in data.items():
        if isinstance(value, dict):
            group = f.create_group(key)
            write_recursive_hdf5(group, value)
        elif isinstance(value, h5py.SoftLink):
            f[key] = value
        elif value is not None:
            # Base case
            f.create_dataset(key, data=value)


def write_hdf5(data, output_path):
    with h5py.File(output_path, "w") as f:
        write_recursive_hdf5(f, data)


class RelScaleNetCached:

    def __init__(self, model_path, device, verbose=False, num_workers=4, resize=False, max_dim=640, batch_size=1028, double_precision=False):
        self.device = device
        self.model = self._load_model(model_path, device)
        if double_precision:
            self.model.double()
        self.cache = {}
        self.resize = resize
        self.max_dim = max_dim
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_precision = double_precision

    def _load_model(self, path, device):
        model = RelScaleNet()
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        return model

    def is_cached(self, cache_key):
        return cache_key in self.cache
    
    def predict_batch(self, X):
        X = X.to(self.device)
        t1 = time.time()
        pred_rel_scale = self.model(X)
        t2 = time.time()
        inference_time = t2 - t1
        if self.verbose:
            print("Inference time:", inference_time)
        return pred_rel_scale

    def load_from_pickle(self, cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def save_to_pickle(self, cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def create_h5_from_pickle(self, cache_path):
        pickled_path = cache_path.replace('.h5', '.p')
        self.load_from_h5(cache_path)
        if os.path.exists(pickled_path):
            print('Loading data from pickle.')
            with open(pickled_path, 'rb') as f:
                pickled_cache = pickle.load(f)
            write_recursive_hdf5(self.cache, pickled_cache)

    def load_from_h5(self, cache_path, store_in_memory=True):
        if store_in_memory:
            # Load the entire h5 file into memory
            h5_args = {'driver': 'core', 'backing_store': True}
        else:
            # Load the h5 file from disk as needed
            h5_args = {'driver': None}
        self.cache = h5py.File(cache_path, 'a', **h5_args)

    def save_to_h5(self, cache_path=None):
        self.cache.flush()
        # write_recursive_hdf5(self.cache, cache_path)

    def predict_image_pair(self, im1_path, im2_path, x1, x2, cache_key=None):
        if cache_key is None:
            cache_key = f'{im1_path}_{im2_path}'
            
        # Check if result is in the cache
        if cache_key in self.cache and 'pred' in self.cache[cache_key]:
            if self.verbose:
                print(f'Retrieving cached scale for {cache_key}')
            pair = self.cache[cache_key]
            assert(pair['x1'].shape[0] == x1.shape[0] and pair['x2'].shape[0] == x2.shape[0])
            return pair['pred']

        # Everything below runs if and only if pred was not found in cache for key
        else:
            if self.verbose:
                print(f'Predicting relative scale for {cache_key}')
            if cache_key not in self.cache:
                if isinstance(self.cache, dict):
                    self.cache[cache_key] = {}
                else:
                    # Using h5 cache
                    self.cache.create_group(cache_key)

        im1, scale1 = load_image(im1_path, self.resize, self.max_dim)
        im2, scale2 = load_image(im2_path, self.resize, self.max_dim)

        # Write coordinates to cache (unless they already exist)
        if isinstance(self.cache, dict):
            self.cache[cache_key] =  {'x1': x1, 'x2': x2}
        elif not ('x1' in self.cache[cache_key] and 'x2' in self.cache[cache_key]):
            # Using h5 cache
            write_recursive_hdf5(self.cache[cache_key], {'x1': x1, 'x2': x2})

        x1 = x1.copy() * scale1
        x2 = x2.copy() * scale2

        patches1 = extract_patches(im1, x1)
        patches2 = extract_patches(im2, x2)

        N = len(x1)
        rel_scales = []
        for i in range(0, N, self.batch_size):
            start_idx = i
            end_idx = min(N, start_idx+self.batch_size)
            sz = end_idx - start_idx
            # Merge patches into one batch
            p3 = torch.ones((sz,6,64,64))
            for k, (p1,p2) in enumerate(zip(patches1[start_idx:end_idx],patches2[start_idx:end_idx])):
                p3[k] = torch.cat((p1,p2),dim=1)
            if self.double_precision:
                p3 = p3.double()
            
            with torch.no_grad():
                preds = list(self.predict_batch(p3).cpu().numpy())
                rel_scales += preds

        pred = scale2 / scale1 * np.array(rel_scales)

        self.cache[cache_key]['pred'] = pred

        return pred