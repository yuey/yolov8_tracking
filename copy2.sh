# replace with the root containing the decompressed train/, test/, challenge/ folders
DATA_DIR=~/challenge_data/tracking-2023

DST_DIR=val_utils/data/SNMOT/test_small_2
mkdir -p ${DST_DIR}
cp -r ${DATA_DIR}/test/SNMOT-197 ${DST_DIR}  # 116, 117, 118, 119 match in the test set
# HOTA doesn't like the last 3 columns being -1
ls ${DST_DIR}/SNMOT-197/gt/gt.txt | xargs -I% sh -c "sed 's/,-1,-1,-1$/,1,1,1/' % > %_; mv % %.orig; mv %_ %"
cp -r ${DATA_DIR}/test/SNMOT-198 ${DST_DIR}  # 116, 117, 118, 119 match in the test set
# HOTA doesn't like the last 3 columns being -1
ls ${DST_DIR}/SNMOT-198/gt/gt.txt | xargs -I% sh -c "sed 's/,-1,-1,-1$/,1,1,1/' % > %_; mv % %.orig; mv %_ %"
cp -r ${DATA_DIR}/test/SNMOT-199 ${DST_DIR}  # 116, 117, 118, 119 match in the test set
# HOTA doesn't like the last 3 columns being -1
ls ${DST_DIR}/SNMOT-199/gt/gt.txt | xargs -I% sh -c "sed 's/,-1,-1,-1$/,1,1,1/' % > %_; mv % %.orig; mv %_ %"
cp -r ${DATA_DIR}/test/SNMOT-200 ${DST_DIR}  # 116, 117, 118, 119 match in the test set
# HOTA doesn't like the last 3 columns being -1
ls ${DST_DIR}/SNMOT-200/gt/gt.txt | xargs -I% sh -c "sed 's/,-1,-1,-1$/,1,1,1/' % > %_; mv % %.orig; mv %_ %"