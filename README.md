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

# for example
python3 predict.py --image corn_test/frame000100.png
```
__Note__: `utils.py`, `transforms.py`, `coco_eval.py`, `coco_utils.py`, `engine.py` contains helper functions used during training process, and they are adopted from PyTorch Repo.

## Pretrained weight

If you want to use a pretrained weight, put the weight in output/ directory and run prediction right away. [10 epochs](https://drive.google.com/file/d/1kYnjmNrVMV7-127w0UiH3SK21of9iqEM/view?usp=sharing), [100 epochs](https://drive.google.com/file/d/1RxDcqj12VF9XcBvB9UMNdOocL6XfuJEP/view?usp=sharing)

## tracking using optical flow
Tested with opencv 4.2.0 with python 3.8

If you face errors with your opencv, comment out the followings:
    cv2.namedWindow, cv2.imshow, cv2.waitKey

This should not affect the functionality of the script.

`--frame_out` decides how often you want to run detection. If it is 200, you will run detection every 200 frame. For our initial test with the corn data, 200 works best.

```shell
python3 tracking_opticalflow.py --video VideoFile.avi --output OutputFile.avi --frame_count NumberOfFrame

# for example
python3 tracking_opticalflow.py --video 8-6_10-8.avi --output results.avi --frame_count 200
