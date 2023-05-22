import SoccerNet
import os
import numpy as np
import pandas as pd
import re
import yaml

def main():
    # base settings
    full_path = "/home/ubuntu/challenge_data/tracking-2023/"
    train_path = os.path.join(full_path, 'train')
    valid_path = os.path.join(full_path, 'test')

    train_dirs = [os.path.join(train_path, x) for x in os.listdir(train_path)]
    val_dirs = [os.path.join(valid_path, x) for x in os.listdir(valid_path)]

    # soccernet data setting
    split_dirs = {
        'train':train_dirs,
        'valid':val_dirs,
    }

    # ４ types of labels are used.
    labels = ['ball', 'player', 'referee', 'goalkeepers']
    label_dict = {'ball': 0, 'player': 1, 'referee': 2, 'goalkeeper': 3, 'goalkeepers': 3}

    # yolo setting
    yolo_base = "/home/ubuntu/cs231n/data/yolo_base"

    #(1) image file path
    yolo_train_img_dir = f'{yolo_base}/images/train'
    yolo_valid_img_dir = f'{yolo_base}/images/valid'

    #(2) label file path
    yolo_train_label_dir = f'{yolo_base}/labels/train'
    yolo_valid_label_dir = f'{yolo_base}/labels/valid'

    #(3) config file path
    yaml_file = f'{yolo_base}/data.yaml'

    os.makedirs(yolo_train_img_dir, exist_ok=True)
    os.makedirs(yolo_valid_img_dir, exist_ok=True)
    os.makedirs(yolo_train_label_dir, exist_ok=True)
    os.makedirs(yolo_valid_label_dir, exist_ok=True)

    # convert from x,y,w,h to yolo format
    def get_yolo_format_bbox(img_w, img_h, box):
        w = box[2]
        h = box[3]
        xc = box[0] + int(np.round(w/2))
        yc = box[1] + int(np.round(h/2))
        box = [xc/img_w, yc/img_h, w/img_w, h/img_h]
        box = [f"{i:.4g}" for i in box]
        return box
        
    # get SoccerNet label info 
    def get_info(info):
        results = []
        for line in open(info):
            m = re.match('trackletID_(\d+)= (\S*).*', line.replace(';', ' '))
            if m:
                if m.group(2) not in label_dict:
                    #print('bad label:', m.group(2))
                    continue 
                results.append([m.group(1), m.group(2)])
        return pd.DataFrame(results, columns=['id','lbl']).set_index('id').to_dict()['lbl']

    # make image simlink and label files
    for split in split_dirs:
        if split == 'train':
            yolo_img_dir = yolo_train_img_dir
            yolo_label_dir = yolo_train_label_dir
        else:
            yolo_img_dir = yolo_valid_img_dir
            yolo_label_dir = yolo_valid_label_dir
        
            
        for this_dir in split_dirs[split]:
            print('this_dir:',this_dir)
            video = this_dir.split('/')[-1]
            info = this_dir + '/gameinfo.ini'
            det = this_dir + '/gt/gt.txt'
            info_dict = get_info(info)
            det_df = pd.read_csv(det, names=['frame','player','x','y','w','h','f1','f2','f3','f4'], usecols=['frame','player','x','y','w','h'])
            det_df['label'] = det_df.player.astype(str).map(info_dict)
            det_df['label_id'] = det_df['label'].map(label_dict)
            # check
            ng_list = list(det_df[det_df.label_id.isnull()].label.unique())
            if len(ng_list)>0:
                #print('ng_list:',ng_list, det_df.dropna().shape, det_df.shape)
                det_df = det_df.dropna()
            for grp, grp_df in det_df.groupby('frame'):
                frame = f'{grp:06}'
                img_file = f'{this_dir}/img1/{frame}.jpg'
                dst_file = f'{yolo_img_dir}/{video}_{frame}.jpg'
                if not os.path.exists(dst_file):
                    os.symlink(img_file, dst_file)
                    #print(img_file)
                #img = cv2.imread(dst_file)
                #height, width, _ = img.shape 
                height, width = 1080, 1920
                bboxes = []
                for arr in grp_df[['x', 'y', 'w', 'h', 'label_id']].values:
                    box = get_yolo_format_bbox(width, height, arr[:4])
                    bboxes.append([arr[4]]+box)
                file_name = f'{yolo_label_dir}/{video}_{frame}.txt'
                with open(file_name, 'w') as f:
                    for i, bbox in enumerate(bboxes):
                        bbox = [str(i) for i in bbox]
                        str_bbox = ' '.join(bbox)
                        f.write(str_bbox)
                        f.write('\n')

    # Dump config file
    data_yaml = dict(
        train = yolo_train_img_dir,
        val = yolo_valid_img_dir,
        nc = 4,
        names = labels
    )

    with open(yaml_file, 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

if __name__ == "__main__":
    main()