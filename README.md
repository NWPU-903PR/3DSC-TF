# 3DSC-TF: Classification of Alzheimer's Disease by Jointing 3D Depthwise Separable Convolutional Neural Network and Transformer

3DSC-TF is a deep learning framework that assists doctors in diagnosing Alzheimer's disease. It utilizes a combination of depthwise separable convolution and Transformer to extract features from sMRI images, while also providing a certain level of interpretability.

## Requirements
The project is written in Python 3.5 and all experiments were conducted on the NVIDIA GTX Titan V GPU. For the faster training process, training on a GPU is necessary, but a standard computer without GPU also works (consuming much more training time). 

All implementations of 3DSC-TF and the CNN-based baselines were based on PyTorch. 3DSC-TF requires the following dependencies:

- python == 3.5.2
- numpy == 1.19.5
- pandas == 1.1.5
- nibable == 3.2.1
- pytorch == 1.10.1


## Reproducibility
- Train: The scripts of `3DSC-TF_train.py` are used for training the network.
- Test: The scripts of `3DSC-TF_train.py` are used for testing the network performance.
- Visualization: The scripts of `visual_patch.py` are used for interpretability analysis.




