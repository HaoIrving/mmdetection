cd ..


# test
# python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_300.pth #--fuse-conv-bn
python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_240.pth
python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_240.pth
python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_240.pth
python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_240.pth
python tools/test.py configs/sardet/faster_rcnn_r50_c4_1x_sarship.py work_dirs/faster_rcnn_r50_c4_1x_sarship/epoch_240.pth
# python tools/test.py configs/sardet/ssd_512_1x_sarship.py work_dirs/ssd_512_1x_sarship/epoch_290.pth
# python tools/test.py configs/sardet/cascade_r101_fpn_dcn_1x_sarship.py work_dirs/cascade_r101_fpn_dcn_1x_sarship/epoch_90.pth


# log
# scp -i qiaohong~qiaohong_sar -P 25392 root@172.18.41.31:/root/data/SAR/mmdetection/work_dirs/ssd_512_1x_sarship/20201221_103501.log.json /home/sun/projects/mmdetection/work_dirs
# scp -i qiaohong~qiaohong_sar -P 25392 root@172.18.41.31:/root/data/SAR/mmdetection/work_dirs/cascade_r101_dcn_1x_sarship/20201222_013502.log.json /home/sun/projects/mmdetection/work_dirs
# python ./tools/analyze_logs.py plot_curve work_dirs/ssd_512_1x_sarship/20201230_141152.log.json  --keys bbox_mAP_50 bbox_mAP_s bbox_mAP_m bbox_mAP_l --legend bbox_mAP_50 bbox_mAP_s bbox_mAP_m bbox_mAP_l