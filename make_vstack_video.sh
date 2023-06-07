# ffmpeg -f image2 \
# -i runs/gt/SNMOT-116/%06d.jpg \
# -i runs/detr/SNMOT-116_newvis/%06d.jpg \
# -i runs/gddino/SNMOT-116_newvis/%06d.jpg \
# -filter_complex \
# "[0:v]scale=-1:720[v0]; \
# [1:v]scale=-1:720[v1]; \
# [2:v]scale=-1:720[v2]; \
# [v0][v1][v2]vstack=inputs=3[v]" -map "[v]" \
# videos/vstacked_gt_detr_gddino.mp4

# ffmpeg -f image2 \
# -i runs/gt/SNMOT-116/%06d.jpg \
# -i runs/val/original_yolov8x/SNMOT-116_newvis/%06d.jpg \
# -filter_complex \
# "[0:v]scale=-1:720[v0]; \
# [1:v]scale=-1:720[v1]; \
# [v0][v1]vstack=inputs=2[v]" -map "[v]" \
# videos/vstacked_rethink_116.mp4

# ffmpeg -f image2 \
# -i runs/gt/SNMOT-116/%06d.jpg \
# -i runs/tsv_test/SNMOT-116_newvis/%06d.jpg \
# -filter_complex \
# "[0:v]scale=-1:720[v0]; \
# [1:v]scale=-1:720[v1]; \
# [v0][v1]vstack=inputs=2[v]" -map "[v]" \
# videos/vstacked_gt_transformer_116.mp4

ffmpeg -f image2 \
-i runs/gt/SNMOT-116/%06d.jpg \
-i runs/detr/SNMOT-116_newvis/%06d.jpg \
-i runs/grounding-dino/SNMOT-116_newvis/%06d.jpg \
-i runs/faster-rcnn/SNMOT-116_newvis/%06d.jpg \
-i runs/mask-rcnn/SNMOT-116_newvis/%06d.jpg \
-i runs/yolo_base/SNMOT-116_newvis/%06d.jpg \
-i runs/yolo_tuned/SNMOT-116_newvis/%06d.jpg \
-filter_complex \
"[0:v]scale=-1:720,pad=3840:720:(ow-iw)/2:(oh-ih)/2[v0]; \
[1:v]scale=-1:720[v1]; \
[2:v]scale=-1:720[v2]; \
[3:v]scale=-1:720[v3]; \
[v1][v2][v3]hstack=inputs=3[vg1]; \
[4:v]scale=-1:720[v4]; \
[5:v]scale=-1:720[v5]; \
[6:v]scale=-1:720[v6]; \
[v4][v5][v6]hstack=inputs=3[vg2]; \
[v0][vg1][vg2]vstack=inputs=3[v]" -map "[v]" \
videos/sstacked_gt_detr_ggdino_fasterrcnn__gt_maskrcnn_yolobase_yolotuned.mp4