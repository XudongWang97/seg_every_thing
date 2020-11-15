## Learning to Segment Every Thing

### Training

1. Stage 1 (bbox training on 3k VG classes):

```
python tools/train_net.py --cfg configs/hardhat_uniform/eval_sw/stage1_e2e_fast_rcnn_R-50-FPN_1x_1im.yaml
```

2. Weights "surgery" 1: convert 3k VG detection weights to 80 COCO detection weights: 

```
python tools/vg3k_training/convert_vg3k_det_to_coco.py --input_model train/vg3k_cocoaligned_train_hardhat:hardhat_uniform_train_vg3k/generalized_rcnn/model_final.pkl --output train/vg3k2coco/vg3k2coco_det.pkl
```

3. Stage 2 (mask training on 80 COCO classes):

```
python tools/train_net.py --cfg configs/hardhat_uniform/eval_sw/stage2_cocomask_clsbox_2_layer_mlp_nograd.yaml
```

4. Weights "surgery" 2: convert 80 COCO detection weights back to 3k VG detection weights:

```
python tools/vg3k_training/convert_coco_seg_to_vg3k.py --input_model train/coco_2014_train:coco_2014_valminusminival/mask_rcnn_frozen_features/model_final.pkl --output_model train/coco2vg3k/coco2vg3k_det.pkl
```

### Inference

```
python tools/infer_simple.py --cfg configs/hardhat_uniform/eval_sw/runtest_clsbox_2_layer_mlp_nograd.yaml --output-dir visualizations-hardhat_uniform --image-ext jpg --thresh 0.5 --use-vg3k --wts train/coco2vg3k/coco2vg3k_det.pkl demo_hardhat
```