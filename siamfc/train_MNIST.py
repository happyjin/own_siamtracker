import os
import lmdb
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from siamfc.config import config
from siamfc.siamnet import SiameseNet
from siamfc.dataset import TrackerDataset
from siamfc.custom_transforms import ToTensor, RandomStretch, CenterCrop
import matplotlib.pyplot as plt

torch.manual_seed(config.seed)

def train(data_dir, cuda_num):
    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(all_videos,
                                                  test_size=1 - config.train_ratio,
                                                  random_state=config.seed)

    # define transforms
    train_z_transforms = transforms.Compose([
        RandomStretch(config.scale_resize),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # open lmdb
    db = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = TrackerDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)
    valid_dataset = TrackerDataset(db, valid_videos, data_dir, train_z_transforms, train_x_transforms)

    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=config.valid_num_workers, drop_last=True)

    # create summary writer
    #if not os.path.exists(config.log_dir):
    #   os.mkdir(config.log_dir)
    #summary_writer = SummaryWriter(config.log_dir)

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
    else:
        dtype = torch.FloatTensor

    model = SiameseNet(dtype)
    if use_gpu:
        model = model.cuda()

    model.init_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(config.epoch):
        model.train()
        for i, data in enumerate(trainloader):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.type(dtype)), Variable(instance_imgs.type(dtype))
            optimizer.zero_grad()
            scores, target_models = model.forward((exemplar_var, instance_var)) # (b,t,c,h,w)
            loss = model.total_loss(scores, target_models)
            print(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            step = epoch * len(trainloader) + i
            #summary_writer.add_scalar('train/loss', loss.data, step)

        if epoch % config.save_interval == 0:
            torch.save(model.cpu().state_dict(), "./models/MNIST_siamfc_{}.pth".format(epoch + 1))

        if use_gpu:
            model = model.cuda()


if __name__ == '__main__':
    data_dir = '/Users/lijin/Documents/GitHub/attention_siamfc/DATASET/Moving_MNIST_CURATION'
    train(data_dir, cuda_num=1)