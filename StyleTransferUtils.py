import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL

import numpy as np

SQUEEZENET_MEAN = [0.485, 0.456, 0.406]
SQUEEZENET_STD = [0.229, 0.224, 0.225]

dtype = torch.FloatTensor

#Style, content loss functions and helpers
def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    content_loss = content_weight*torch.sum((content_original-content_current)**2)
    
    return content_loss

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    style_loss = 0.0
    
    for i,layer in enumerate(style_layers):
        G = gram_matrix(feats[layer])
        style_loss += style_weights[i]*torch.sum((G-style_targets[i])**2)

    return style_loss

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    horizontal_shift_1 = img[:,:,:,1:]
    horizontal_shift_2 = img[:,:,:,:-1]
    
    vertical_shift_1 = img[:,:,1:,:]
    vertical_shift_2 = img[:,:,:-1,:]
    
    tv_loss = tv_weight*(torch.sum((horizontal_shift_1-horizontal_shift_2)**2)+torch.sum((vertical_shift_1-vertical_shift_2)**2))
    
    return tv_loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """

    N,C,H,W = features.shape
    
    gram_features = torch.reshape(features,(N,C,-1))
    
    gram = torch.matmul(gram_features,gram_features.permute(0,2,1))
    
    if normalize == True:
        gram /= (H*W*C)
    
    return gram

###################################################################################################################
# Methods below copied from Stanford CS231n's assignment #3 https://cs231n.github.io/assignments2020/assignment3/ #
# For converting jpg images to pytorch tensors and back
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
        T.Normalize(mean=SQUEEZENET_MEAN,
                    std=SQUEEZENET_STD),
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
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN], std=[1, 1, 1]),
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

#returns a list of feature values at each leayer of the model, cnn, when given an input x
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i);
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

#get image and CNN features from input image
def features_from_img(imgpath, imgsize, cnn):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(dtype)
    return extract_features(img_var, cnn), img_var