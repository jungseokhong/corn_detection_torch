# Faster R-CNN for corn detection with Pytorch
Written based on https://github.com/haochen23/fine-tune-MaskRcnn

Using Pytorch `1.7.1+cu110`

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

## Pretrained weight

If you want to use a pretrained weight, put the weight in output/ directory and run prediction right away.[link](https://drive.google.com/file/d/1yf2qIMwuIY4Q9DvlYxcVmpbvJSuZ4QVt/view?usp=sharing)