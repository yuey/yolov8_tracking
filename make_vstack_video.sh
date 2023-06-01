ffmpeg -f image2 \
-i /home/ubuntu/dev/yolov8_tracking/runs/gt/SNMOT-118/%06d.jpg \
-i runs/val/withball_yolov8x/SNMOT-118/%06d.jpg \
-i runs/val/rerun_0527/SNMOT-118/%06d.jpg \
-filter_complex \
"[0:v]scale=-1:720[v0]; \
[1:v]scale=-1:720[v1]; \
[2:v]scale=-1:720[v2]; \
[v0][v1][v2]vstack=inputs=3[v]" -map "[v]" \
videos/vstacked_118.mp4

# ffmpeg -f image2 \
# -i runs/gt/SNMOT-116/%06d.jpg \
# -i runs/val/original_yolov8x/SNMOT-116_newvis/%06d.jpg \
# -filter_complex \
# "[0:v]scale=-1:720[v0]; \
# [1:v]scale=-1:720[v1]; \
# [v0][v1]vstack=inputs=2[v]" -map "[v]" \
# videos/vstacked_rethink_116.mp4