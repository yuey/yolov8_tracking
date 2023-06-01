"""
Author: Roman Solovyev, IPPM RAS
URL: https://github.com/ZFTurbo
"""

from map_boxes import mean_average_precision_for_boxes
import numpy as np
import pandas as pd

IMG_WIDTH, IMG_HEIGHT = 1920, 1080

SAMPLES = [
    "SNMOT-116", "SNMOT-117", "SNMOT-118", "SNMOT-119", "SNMOT-120", "SNMOT-121", "SNMOT-122", "SNMOT-123", "SNMOT-124", "SNMOT-125",
    "SNMOT-126", "SNMOT-127", "SNMOT-128", "SNMOT-129", "SNMOT-130", "SNMOT-131", "SNMOT-132", "SNMOT-133", "SNMOT-134", "SNMOT-135",
    "SNMOT-136", "SNMOT-137", "SNMOT-138", "SNMOT-139", "SNMOT-140", "SNMOT-141", "SNMOT-142", "SNMOT-143", "SNMOT-144", "SNMOT-145",
    "SNMOT-146", "SNMOT-147", "SNMOT-148", "SNMOT-149", "SNMOT-150", "SNMOT-187", "SNMOT-188", "SNMOT-189", "SNMOT-190", "SNMOT-191",
    "SNMOT-192", "SNMOT-193", "SNMOT-194", "SNMOT-195", "SNMOT-196", "SNMOT-197", "SNMOT-198", "SNMOT-199", "SNMOT-200",
]


PRINT_DET_BALL = False


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
    det_folder = '/home/ubuntu/dev/yolov8_tracking/runs/detr_baseline_all_test_metrics'

    anns, dets = [], []
    for sample in SAMPLES:

        ##########
        ann = pd.read_csv(f'{gt_folder}/test/{sample}/gt/gt.txt', header=None)
        ann.iloc[:, 0] = ann.iloc[:, 0].apply(lambda x: f'{sample}:{x:03d}')
        # parse gameinfo (trackletID to our class string here)
        ann_class_column = 1
        gameinfo_dict = parse_gameinfo(f'{gt_folder}/test/{sample}/gameinfo.ini')
        def map_numbers_to_strings(number):
            return 'ball' if gameinfo_dict[number] == 'ball' else 'person'
        ann.iloc[:, ann_class_column] = ann.iloc[:, ann_class_column].astype(str).apply(map_numbers_to_strings)
        ann = ann.values[:, [0, 1, 2, 3, 4, 5]]

        print(ann)
        # for i in ann.tolist():
        #     if i[1] == 'ball':
        #         print(i)
        ##########
        anns.append(ann)

        ##########
        det = pd.read_csv(f'{det_folder}/{sample}/deter_base_bbox.tsv', sep='\t', header=None)
        det.iloc[:, 0] = det.iloc[:, 0].apply(lambda x: f'{sample}:{x:03d}')
        class_column = 7  # 6 is enum, 7 is string
        det.iloc[:, class_column] = det.iloc[:, class_column].replace('sports ball', 'ball')
        det = det.values[:, [0, class_column, 5, 1, 2, 3, 4]]
        print(det)
        if PRINT_DET_BALL:
            for i in det.tolist():
                if i[1] == 'ball':
                    print(i)
        ##########
        dets.append(det)

    anns = np.concatenate(anns)
    dets = np.concatenate(dets)

    xywh_to_norm_xxyy(anns)
    xywh_to_norm_xxyy(dets)

    # ann:'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'
    # det:'ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'
    mean_ap, average_precisions = mean_average_precision_for_boxes(anns, dets, iou_threshold=0.5, verbose=True)
    print(mean_ap)
    print(average_precisions)
