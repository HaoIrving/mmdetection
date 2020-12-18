cd ..

bash ./tools/dist_train.sh \
    ./configs/sardet/faster_rcnn_r50_c4_1x_sarship.py  \
    4 \

# benchmark fps
# python tools/benchmark.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_300.pth --fuse-conv-bn

# test
# python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_300.pth #--fuse-conv-bn