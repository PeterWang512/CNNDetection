#!/bin/bash
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007 &
wait $(jobs -p)

7z x progan_train.7z.001
rm progan_train.7z.*
unzip progan_train.zip
rm progan_train.zip
