#!/bin/sh

cd /script

python3 setup.py build_ext --inplace

if [ $gpu ]; then
  ./flow --imgdir $input_dir --output $output_dir --model cfg/yolov2.cfg --load bin/yolov2.weights --threshold 0.05 --gpu 1.0 --json
else
  ./flow --imgdir $input_dir --output $output_dir --model cfg/yolov2.cfg --load bin/yolov2.weights --threshold 0.05 --json
fi
