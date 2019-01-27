import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from siamfc.siamnet import SiameseNet
from siamfc.config import config
from siamfc.custom_transforms import ToTensor
from siamfc.utils import get_exemplar_image, get_pyramid_instance_image


class SiamTracker:
    def __init__(self, model_path):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        self.dtype = dtype
        self.model = SiameseNet(dtype)
        self.model.load_state_dict(torch.load(model_path))
        if use_gpu:
            self.model = self.model.cuda()
        self.model.eval()

        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def _cosine_window(self, size): #????what is size??? How to understand window function in this paper?
        """
        get the cosine windows
        :param size:
        :return:
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """
        initialize the siam tracker
        :param frame: an RGB image
        :param bbox: [xmin, ymin, width, height]
            one-based bounding box
        :return:
        """
        self.bbox = (bbox[0] - 1, bbox[1] - 1, bbox[0] - 1 + bbox[2], bbox[1] - 1 + bbox[3])  # (xmin,ymin,xmax,ymax)
        self.target_center = np.array([bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2])  # [center_x, center_y]
        self.target_sz = np.array([bbox[2], bbox[3]])  # (width, height)

        # get exemplar image
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox, config.exemplar_size, config.context_amount,
                                                        self.img_mean)
        #plt.imshow(exemplar_img)
        #plt.show()

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        exemplar_img_var = Variable(exemplar_img.type(self.dtype))
        self.model.forward((exemplar_img_var, None)) # transfer gt exemplar namely gt exemplar at 1st frame into SiameseNet

        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1 #=[0.975,1,0.975] choose center position of penalty array as 1 other parts assign to penalty value

        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz # interpolation size by upsampling_
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))

        # create scales for pyramid
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale / 2) - config.num_scale,
                                                     np.floor(config.num_scale / 2) + 1)

        # create s_x
        self.s_x = s_z + (config.instance_size - config.exemplar_size) / scale_z

        # arbitrary scale saturation ###????? why saturation???
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """
        track object based on the previous frame
        :param frame: an RGB image
        :return:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(frame, self.target_center, config.instance_size, size_x_scales, self.img_mean)

        instance_imgs = torch.cat([self.transforms(x)[None, :, :, :] for x in pyramid], dim=0)
        instance_imgs_var = Variable(instance_imgs.type(self.dtype))
        response_maps = self.model.forward((None, instance_imgs_var))
        response_maps = response_maps.data.cpu().numpy().squeeze()

        # visualize the response map
        #plt.imshow(response_maps[0])
        #plt.show()

        response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC) for x in response_maps]
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty

        # penality scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        # make the response map sum to 1 namely normalization
        response_map -= response_map.min()
        response_map /= response_map.sum()

        # apply windowing
        response_map = (1 - config.window_influence) * response_map + config.window_influence * self.cosine_window
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)

        # Convert the crop-relative coordinates to frame coordinates
        # displacement from the center in instance to final representation
        p = np.array([max_c, max_r])  # position of max response in response_maps_up
        center = (self.interp_response_sz - 1) / 2  # center of response_maps_up
        disp_response_interp = p - center  # displacement in the interpolation response map
        # displacement in instance input
        disp_instance_input = disp_response_interp * config.total_stride / config.response_up_stride  ###??????what want to compute??
        # displacement in instance frame
        scale = self.scales[scale_idx]
        disp_instance_frame = disp_instance_input * (self.s_x * scale) / config.instance_size
        # position within frame in frame coordinates
        self.target_center += disp_instance_frame
        # scale damping and saturation #????? what is damping and saturation?
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz
        bbox = (self.target_center[0] - self.target_sz[0] / 2 + 1,  # xmin conver to 1-based
                self.target_center[1] - self.target_sz[1] / 2 + 1,  # ymin
                self.target_center[0] + self.target_sz[0] / 2 + 1,  # xmax
                self.target_center[1] + self.target_sz[1] / 2 + 1)  # ymax
        return bbox