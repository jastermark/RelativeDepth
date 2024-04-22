import os
import numpy as np
import h5py


def h5_to_camera_dict(data):
    camera_dict = {}
    camera_dict['model'] = data['model'].asstr()[0]
    camera_dict['width'] = int(data['width'][0])
    camera_dict['height'] = int(data['height'][0])
    camera_dict['params'] =data['params'][:]
    return camera_dict

def camera_dict_to_calib_matrix(cam):
    if cam['model'] == 'PINHOLE':
        p = cam['params']
        return np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
    else:
        raise Exception('nyi model in camera_dict_to_calib_matrix')
        

class ImagePair:
    def __init__(self, image_dir, data) -> None:
        self.data = data
        self.image_dir = image_dir
        self.key = self.data.name.replace('/', '')
 
    def R(self):
        R_gt = self.data['R'][:]
        u,s,vt = np.linalg.svd(R_gt)
        R_gt = u @ vt
        return R_gt

    def t(self, normalize=True):
        t_gt = self.data['t'][:]
        if normalize:
            t_gt = t_gt / np.linalg.norm(t_gt)
        return t_gt

    def relative_pose(self):
        return self.R(), self.t()
    
    def essential_matrix(self):
        R = self.R()
        t = self.t()
        T = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
        E = T @ R
        return E

    def fundamental_matrix(self):
        (K1, K2) = self.calib_matrices()
        E = self.essential_matrix()
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        return F

    def image_names(self):
        return [self.data['name1'].asstr()[0], self.data['name2'].asstr()[0]]

    def image_paths(self):
        return [os.path.join(self.image_dir, name) for name in self.image_names()]
    
    def matches(self):
        x1 = self.data['x1'][:]
        x2 = self.data['x2'][:]
        return (x1, x2)

    def cameras(self):
        return (h5_to_camera_dict(self.data['camera1']), h5_to_camera_dict(self.data['camera2']))

    def calib_matrices(self):
        return (camera_dict_to_calib_matrix(h5_to_camera_dict(self.data['camera1'])),
                camera_dict_to_calib_matrix(h5_to_camera_dict(self.data['camera2'])))

        

class EvaluationDataset:

    def __init__(self, h5_path, im_path) -> None:
        assert os.path.exists(h5_path), "Image directory not found"
        assert os.path.exists(im_path), "Ground truth file not found"
        self.data = h5py.File(h5_path, 'r')
        self.image_dir = im_path
        self.pairs = list(self.data.keys())

    
    def __iter__(self):
        for i in range(len(self.pairs)):
            yield self[i]
    
    def __len__(self):
        return len(self.data.items())
    
    def __getitem__(self, idx):
        return ImagePair(self.image_dir, self.data[self.pairs[idx]])
