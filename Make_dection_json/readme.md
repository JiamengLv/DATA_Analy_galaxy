


1.mmdet框架下的attentionmap的可视化，主要涉及两个文件：

pip install gra_cam==1.4.0 

1.1 将vis_cam.py文件放到demo这个目录中

1.2 将det_cam_visualizer.py文件放到mmdet\utils\目录中

1.3 命令：

FeatmapAM method
python demo/vis_cam.py demo/demo.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

EigenCAM method
python demo/vis_cam.py demo/demo.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method eigencam

AblationCAM method
python demo/vis_cam.py demo/demo.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method ablationcam

AblationCAM method and save img
python demo/vis_cam.py demo/demo.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method ablationcam --out-dir save_dir

GradCAM
python demo/vis_cam.py demo/demo.jpg configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --method gradcam
