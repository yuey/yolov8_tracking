# replace with the root containing the decompressed train/, test/, challenge/ folders
DATA_DIR=~/challenge_data/tracking-2023

DST_DIR=val_utils/data/SNMOT/test_small
mkdir -p ${DST_DIR}
cp -r ${DATA_DIR}/test/SNMOT-11* ${DST_DIR}  # 116, 117, 118, 119 match in the test set
# HOTA doesn't like the last 3 columns being -1
ls ${DST_DIR}/SNMOT-11*/gt/gt.txt | xargs -I% sh -c "sed 's/,-1,-1,-1$/,1,1,1/' % > %_; mv % %.orig; mv %_ %"