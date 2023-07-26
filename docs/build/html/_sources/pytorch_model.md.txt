#  yoloV7を用いた船舶検知 (local)
YOLO v7-tinyをNCS2で動作させたい。ただし、YOLO v7のKeras版がないので、Torch版で進めていく。
最終的にProtoBufferやONNXに変換できればMOでIR形式に変換できるのでOK。

以下を使っていく。
https://github.com/WongKinYiu/yolov7


### Install Torch
TORCHは1.12.0じゃないとエラーがでる。Cudaは11.3がよい。

``` bash
$ pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```


### Install YOLOv7
YOLOをGitからダウンロードして、依存環境をインストール

``` bash
$ git clone https://github.com/WongKinYiu/yolov7
$ cd YOLOv7
$ pip install -r requirements.txt
```

###  Training on YOLOv7
後は良しなにyoolov7-tinyのモデル構造を変更するのと伴にデータセットを作って、学習。
（余力あれば追記）

```bash
$ python train.py --workers 8 --device 0 --batch-size 8 --data data/ship.yaml --img 800 800 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7_ship --hyp data/hyp.scratch.p5.yaml
```

こんな感じになればOK。画像サイズは32の倍数でないといけない。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/394a8b4d-31f9-ccc4-1c4b-fd239ef557b1.png)

学習できていれば停止
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/55d60a44-a979-cebe-65cf-14e27a8d6f22.png)


### Testing on YOLOv7

```bash
$ python detect.py --weights runs/train/yolov7_ship/weights/best.pt --conf 0.8 --img-size 800 --source custom_dataset/images/val/img_99-0.png
```

次のようなログが流れて始めれば成功

```
Namespace(weights=['runs/train/yolov7_ship/weights/best.pt'], source='custom_dataset/images/val/img_99-0.png', img_size=800, conf_thres=0.8, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)
YOLOR ? v0.1-126-g84932d7 torch 1.12.0+cu113 CUDA:0 (NVIDIA TITAN V, 12064.375MB)
                                             CUDA:1 (NVIDIA TITAN V, 12066.875MB)

Fusing layers...
IDetect.fuse
/home/taiyaki/anaconda3/envs/torch/lib/python3.9/site-packages/torch/functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2894.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 208 layers, 6007596 parameters, 0 gradients, 13.0 GFLOPS
 Convert model to Traced-model...
 traced_script_module saved!
 model is traced!

2 ships, Done. (6.8ms) Inference, (1.2ms) NMS
```

推定画像。精度は99%程度
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/046a9c26-f639-52cc-027e-4a88b3056e81.png)

