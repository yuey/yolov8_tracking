"""
Author: Roman Solovyev, IPPM RAS
URL: https://github.com/ZFTurbo
"""

from map_boxes import mean_average_precision_for_boxes
import numpy as np
import pandas as pd

IMG_WIDTH, IMG_HEIGHT = 1920, 1080


def xywh_to_norm_xxyy(box_seq):
    xmin_col = -4
    ymin_col = -3
    width_col = -2
    height_col = -1

    new_xmin_col = -4
    new_xmax_col = -3
    new_ymin_col = -2
    new_ymax_col = -1
    # xmax = width + xmin
    box_seq[:, width_col] += box_seq[:, xmin_col]
    # ymax = height + ymin
    box_seq[:, height_col] += box_seq[:, ymin_col]
    # from xyxy to xxyy
    box_seq[:, [new_xmax_col, new_ymin_col]] = box_seq[:, [width_col, ymin_col]]
    # scale to normalize
    box_seq[:, [new_xmin_col, new_xmax_col]] /= IMG_WIDTH
    box_seq[:, [new_ymin_col, new_ymax_col]] /= IMG_HEIGHT


def parse_gameinfo(gt_ini):
    gameinfo_dict = {}
    with open(gt_ini) as f:
        for line in f:
            line = line.strip()  # remove any leading or trailing whitespace
            if line.startswith('trackletID'):  # only process trackletID lines
                parts = line.split('=')
                if len(parts) == 2:
                    tracklet_id, role_info = parts
                    tracklet_id = tracklet_id.strip().split('_')[-1]  # extract id from 'trackletID_x'
                    role = role_info.split(';')[0].split(' ')[1].strip()  # extract first word of role from the role info
                    gameinfo_dict[tracklet_id] = role
    return gameinfo_dict

if __name__ == '__main__':

    test_xywh= np.array([[192.0, 216.0, 960.0, 324.0]])
    xywh_to_norm_xxyy(test_xywh)
    expected_norm_xxyy= np.array([[0.1, 0.6, 0.2, 0.5]])
    assert((test_xywh == expected_norm_xxyy).all())

    # # Version 1
    # annotations_file = 'example/annotations.csv'
    # detections_file = 'example/detections.csv'
    # mean_ap, average_precisions = mean_average_precision_for_boxes(annotations_file, detections_file)

    # Version 2

    gt_folder = '/home/ubuntu/dev/yolov8_tracking/val_utils/data/SNMOT'
    det_folder = '/home/ubuntu/dev/yolov8_tracking/runs/val/transformer_orig'

    ##########
    ann = pd.read_csv(f'{gt_folder}/test_small/SNMOT-116/gt/gt.txt', header=None)
    # parse gameinfo (trackletID to our class string here)
    ann.iloc[:, 1] = ann.iloc[:, 1].astype(str)
    ann = ann.values[:, [0, 1, 2, 3, 4, 5]]

    gameinfo_dict = parse_gameinfo(f'{gt_folder}/test_small/SNMOT-116/gameinfo.ini')
    def map_numbers_to_strings(number):
        return 'ball' if gameinfo_dict[number] == 'ball' else 'person'
    # vectorize the function
    vfunc = np.vectorize(map_numbers_to_strings)
    # apply it to the second column of the array
    ann[:, 1] = vfunc(ann[:, 1])
    # print(ann)
    # for i in ann.tolist():
    #     if i[1] == 'ball':
    #         print(i)
    ##########

    ##########
    det = pd.read_csv(f'{det_folder}/SNMOT-116/deter_base_bbox.tsv', sep='\t', header=None)
    class_column = 7  # 6 is enum, 7 is string
    det = det.values[:, [0, class_column, 5, 1, 2, 3, 4]]
    print(det)
    for i in ann.tolist():
        if i[1] == 'ball':
            print(i)
    ##########
    ##########

    xywh_to_norm_xxyy(ann)
    xywh_to_norm_xxyy(det)

    # ann:'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'
    # det:'ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, verbose=True)
    print(mean_ap)
    print(average_precisions)
