#!/bin/bash

# Change directory if needed
# cd /path/to/your/project/folder

# Run the Python script
python ../object_dataset.py \
  --video_path "/media/yeongha/MGTEC/KAIST/Benchmark/SMD/VIS_Onshore/Videos/MVI_1469_VIS.avi" \
  --output_path "/home/yeongha/pycharm/blip-vqa-finetune/Data/train_explain_situation/" \
  --total_data_path "/home/yeongha/pycharm/blip-vqa-finetune/Data/total_train.jsonl" \
  --object_path "/media/yeongha/MGTEC/KAIST/Benchmark/SMD/VIS_Onshore/ObjectGT/MVI_1469_VIS_ObjectGT.mat"

