import cv2
import os
import pickle
import hashlib
import numpy as np
from glob import glob
from torch.utils.data.dataset import Dataset
from siamfc.config import config

class TrackerDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]:x[1] for x in self.meta_data}
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]
        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.num_per_epoch is None or not training else config.num_per_epoch

        # data augmentation
        self.max_stretch = config.scale_resize
        self.max_translate = config.max_translate
        self.random_crop_size = config.instance_size

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(weights - center)
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR), gt_w, gt_h

    def RandomCrop(self, sample):
        shape = sample.shape[:2]
        cy_o = (shape[0] - 1) // 2
        cx_o = (shape[1] - 1) // 2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate + 1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate + 1)
        assert abs(cy - cy_o) <= self.max_translate and \
               abs(cx - cx_o) <= self.max_translate
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy

        ymin = cy - self.random_crop_size // 2
        xmin = cx - self.random_crop_size // 2
        ymax = cy + self.random_crop_size // 2 + self.random_crop_size % 2
        xmax = cx + self.random_crop_size // 2 + self.random_crop_size % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch, gt_cx, gt_cy

    def __getitem__(self, idx):
        idx = idx % len(self.video_names)
        video = self.video_names[idx]
        trajs = self.meta_data[video]
        # sample one trajs
        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        exemplar_name = glob(os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x*.jpg".format(trkid)))[0]
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        # sample instance
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)

        # create sample weight, if the sample is far away from center
        # the probability being chosen are high
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx], p=weights)
        instance_name = glob(os.path.join(self.data_dir, video, instance+".{:02d}.x*.jpg".format(trkid)))[0]
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        gt_w, gt_h = float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])

        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)

        exemplar_img = self.z_transforms(exemplar_img)
        # random stretch and crop the instance image
        instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
        instance_img, gt_cx, gt_cy = self.RandomCrop(instance_img)
        instance_img = self.x_transforms(instance_img)
        return exemplar_img, instance_img

    def __len__(self):
        return self.num