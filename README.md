# Faster R-CNN for corn detection with Pytorch
Written based on https://github.com/haochen23/fine-tune-MaskRcnn

## Usage
__Train the model__
```shell
python3 train.py
```
The trained model will be saved in the `output/` with name `faster-rcnn-corn.pt`

__Model Inference__

```shell
python3 predict.py --image path/to/test/image

#for example
python3 predict.py --image corn_test/frame000100.png
```
__Note__: `utils.py`, `transforms.py`, `coco_eval.py`, `coco_utils.py`, `engine.py` contains helper functions used during training process, and they are adopted from PyTorch Repo.
