## Tool recommendation based on EfficientNet V2

Based on screw detection results, screw images are extracted from the full resolution image and passed to an image classification model based on EfficientNet V2 models [1]. Implementation details of EfficientNet V2 family of models can be found in [Keras Applications](https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/efficientnet_v2.py).

### Instructions

Most scripts incorporates an argument parser (argparse module), type "-h" or "--help" in the command line next to the code to get information about the input arguments. Multiple parameters can be changed accordingly including the loading/saving directories and files.

Using default arguments, training and testing full-resolution images should be placed in folders named "train" and "test", respectively.

The following is a description for the included scripts in ToolRecModel directory:


| File name  | Description |
| ------------- | ------------- |
| toolsRecTrainPrep.py   | extracts screw images for training - screws classes  |
| toolsRecTrainPrepNone.py   | create randomized crops from images for training - none class  |
| toolsRecTrainSplit.py   | make the training/validation split  |
| toolsRecTrainAug.py	   | trains the M-4c classification model (data augmentation enabled, based on EfficientNet V2)  |
| toolsRecTestPrep.py	  | extracts screw images for testing - screws classes  |
| toolsRecTestPrepNone.py	   | create randomized crops from images for testing - none class  |
| toolsRecTest.py   | Test the model performance using the test data  |


References:
[1] Tan, M., & Le, Q. (2021, July). Efficientnetv2: Smaller models and faster training. In International Conference on Machine Learning (pp. 10096-10106). PMLR.
