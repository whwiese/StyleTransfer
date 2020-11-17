import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL

import numpy as np

SQUEEZENET_MEAN = [0.485, 0.456, 0.406]
SQUEEZENET_STD = [0.229, 0.224, 0.225]

dtype = torch.FloatTensor

###################################################################################################################
# Methods below copied from Stanford CS231n's assignment #3 https://cs231n.github.io/assignments2020/assignment3/ #
#
###################################################################################################################

def preprocess(img, size=512):
    """ Preprocesses a PIL JPG Image object to become a Pytorch tensor
        that is ready to be used as an input into the CNN model.
        Preprocessing steps:
            1) Resize the image (preserving aspect ratio) until the shortest side is of length `size`.
            2) Convert the PIL Image to a Pytorch Tensor.
            3) Normalize the mean of the image pixel values to be SqueezeNet's expected mean, and
                 the standard deviation to be SqueezeNet's expected std dev.
            4) Add a batch dimension in the first position of the tensor: aka, a tensor of shape
                 (H, W, C) will become -> (1, H, W, C).
    """
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    """ De-processes a Pytorch tensor from the output of the CNN model to become
        a PIL JPG Image that we can display, save, etc.
        De-processing steps:
            1) Remove the batch dimension at the first position by accessing the slice at index 0.
                 A tensor of dims (1, H, W, C) will become -> (H, W, C).
            2) Normalize the standard deviation: multiply each channel of the output tensor by 1/s,
                 scaling the elements back to before scaling by SqueezeNet's standard devs.
                 No change to the mean.
            3) Normalize the mean: subtract the mean (hence the -m) from each channel of the output tensor,
                 centering the elements back to before centering on SqueezeNet's input mean.
                 No change to the std dev.
            4) Rescale all the values in the tensor so that they lie in the interval [0, 1] to prepare for
                 transforming it into image pixel values.
            5) Convert the Pytorch Tensor to a PIL Image.
    """
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    """ A function used internally inside `deprocess`.
        Rescale elements of x linearly to be in the interval [0, 1]
        with the minimum element(s) mapped to 0, and the maximum element(s)
        mapped to 1.
    """
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

#please disregard warnings about initialization
def features_from_img(imgpath, imgsize, cnn):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(dtype)
    return extract_features(img_var, cnn), img_var