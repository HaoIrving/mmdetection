cd ..

bash ./tools/dist_train.sh \
    ./configs/sardet/ssd_512_1x_sarship.py  \
    4 \

git pull origin master

# bash ./tools/dist_train.sh \
#     ./configs/sardet/faster_rcnn_r50_c4_1x_sarship.py  \
#     4 \

bash ./tools/dist_train.sh \
    ./configs/sardet/ssd_512_1x_sarship.py  \
    4 \
   
# --resume-from ./work_dirs/ssd_512_1x_sarship/epoch_265.pth

# benchmark fps
# python tools/benchmark.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_300.pth --fuse-conv-bn

# test
# python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_300.pth #--fuse-conv-bn

# log
# scp -i qiaohong~qiaohong_sar -P 25392 root@172.18.41.31:/root/data/SAR/mmdetection/work_dirs/ssd_512_1x_sarship/20201221_055641.log.json /home/sun/projects/mmdetection/work_dirs
