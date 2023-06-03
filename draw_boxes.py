import csv
import cv2
import os

MODULO_COLORS = [
    (255, 0, 0),  # Red
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (255, 192, 203),  # Pink
    (128, 0, 0),  # Maroon
    (0, 0, 128),  # Navy
    (0, 128, 128),  # Teal
    (128, 255, 0),  # Greenish
    (192, 192, 192),  # Silver
    (0, 0, 0),  # Black
    (105, 105, 105),  # DimGray
    (255, 250, 240),  # FloralWhite
    (128, 0, 255),  # Purpleish
    (210, 105, 30),  # Chocolate
    (70, 130, 180),  # SteelBlue
    (0, 255, 127),  # SpringGreen
]
NUM_FRAMES = 750

for sample in ["SNMOT-116"]:

    IMG_FMT = (
        f"/home/ubuntu/dev/yolov8_tracking/val_utils/data/SNMOT/test_small/{sample}/{sample}/"
        + "{:06d}.jpg"
    )
    # Ground truth
    # LABELS_TXT = f"/home/ubuntu/dev/yolov8_tracking/val_utils/data/SNMOT/test_small/{sample}/gt/gt.txt"
    # OUT_DIR = f"/home/ubuntu/dev/yolov8_tracking/runs/gt/{sample}/"

    # Test run
    # name = "val/original_yolov8x"
    name = "yolo_base"
    LABELS_TXT = f'/home/ubuntu/dev/yolov8_tracking/runs/{name}/labels/{sample}.txt'
    OUT_DIR = f'/home/ubuntu/dev/yolov8_tracking/runs/{name}/{sample}_newvis/'

    YOLO_HACK = name.startswith('yolo_')

    TSV_TXT = None
    TSV_TXT = '/home/ubuntu/dev/yolov8_tracking/runs/yolo_baseline_all_test/SNMOT-116/yolo_base_bbox.tsv'

    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_IMG_FMT = OUT_DIR + "{:06d}.jpg"

    boxes = {}

    if TSV_TXT:
        with open(TSV_TXT) as file:
            reader = csv.reader(file, delimiter="\t")  # Test run results are space-seperated
            for row in reader:
                key = int(row[0])
                values = row

                if key not in boxes:
                    boxes[key] = []

                boxes[key].append(values)
    else:
        with open(LABELS_TXT) as file:
            #reader = csv.reader(file, delimiter=",")  # gt
            reader = csv.reader(file, delimiter=" ")  # Test run results are space-seperated
            for row in reader:
                key = int(row[0])
                values = list(map(int, row[1:]))

                if key not in boxes:
                    boxes[key] = []

                boxes[key].append(values)


    for frame in range(1, NUM_FRAMES + 1):
        img = cv2.imread(IMG_FMT.format(frame))
        cv2.putText(img, f'f{frame}', (10, 1080 - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

        frame_boxes = boxes.get(frame, [])
        for frame_box in frame_boxes:
            if TSV_TXT:
                if frame_box[6] == '1':
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                str_tracklet_id = ''
                #str_tracklet_id = f'{float(frame_box[5]):.3f}'
            else:
                tracklet_id = str(frame_box[0])
                str_tracklet_id = str(tracklet_id)
                color = MODULO_COLORS[tracklet_id % len(MODULO_COLORS)][::-1]
            [xmin, ymin, width, height] = [int(float(a)) for a in frame_box[1:5]]
            if YOLO_HACK:
                xmin = int(xmin - width / 2)
                ymin = int(ymin - height / 2)
            if frame_box[6] == '0':
                print(frame, [xmin, ymin, width, height])
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(
                img, (xmin, ymin), (xmax, ymax), color=color, thickness=2
            )
            h_centering_offset = 8 * len(str_tracklet_id)
            cv2.putText(img, str_tracklet_id, (xmin + width // 2 - h_centering_offset, ymin - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2)
        cv2.imwrite(OUT_IMG_FMT.format(frame), img)
