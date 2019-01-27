import glob
import os
import cv2
import deepdish as dd
from siamfc.tracker import SiamTracker
import sys
sys.path.append(os.getcwd())

def MNIST_generate_bbox(gt):
    """

    :param gt: (float,float)
        (ymin, xmin)
    :return: (xmin,ymin,w,h)
    """
    xmin = gt[1]
    ymin = gt[0]
    width = height = 28
    return [xmin, ymin, width, height]

def MNIST_main(video_dir, gt_path, model_path, save_path):
    gt_array = dd.io.load(gt_path)
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    idx_row = int(filenames[0].split('/')[-2])
    title = video_dir.split('/')[-1]
    tracker = SiamTracker(model_path)

    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = MNIST_generate_bbox(gt_array[idx_row, idx][1]) # choose which digit want to track
            tracker.init(frame, bbox)
            bbox = (bbox[0] - 1, bbox[1] - 1, bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1) #(xmin,ymin,xmax,ymax)
        else:
            bbox = tracker.update(frame) #(xmin,ymin,xmax,ymax)

        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              2)
        gt_bbox = MNIST_generate_bbox(gt_array[idx_row, idx][1])
        gt_bbox = (gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0] - 1), int(gt_bbox[1] - 1)),
                              (int(gt_bbox[2] - 1), int(gt_bbox[3] - 1)),
                              (255, 0, 0),
                              1)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        cv2.imwrite(os.path.join(save_path, str(idx) + '.jpg'), frame)
        cv2.waitKey(30)




if __name__ == '__main__':
    video_dir = '/Users/lijin/Documents/GitHub/attention_siamfc/DATASET/Moving_MNIST/0'
    gt_path = '/Users/lijin/Documents/GitHub/attention_siamfc/DATASET/top_lefts_normal.h5'
    model_path = '/Users/lijin/Documents/GitHub/attention_siamfc/models/MNIST_siamfc_50.pth'
    save_path = '/Users/lijin/Documents/GitHub/attention_siamfc/DATASET/test_results/MNIST/' + os.path.basename(video_dir)
    MNIST_main(video_dir, gt_path, model_path, save_path)
