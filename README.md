#STYLE TRANSFER

An implementation of style transfer based on Stanford CS231n's Spring 2020 assignment #3 (https://cs231n.github.io/assignments2020/assignment3/).
I filled in the loss functions and reproduced style transfer locally to increase my understanding of it, and of PyTorch.

This assignment is based on a 2015 paper by Gatys et al. which can be found here: 
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

Style transfer works by computing a "style loss" and "content loss" at one or more layers in a convolutional neural network to which an image is passed
as input. The style loss is computed using Gram matrices (approximations of covariance matrices) of the style image and the current
generated image. The Gram matrix turns out to be a decent indicatior of the style of a painting, so the difference between these matrices
gives us a measure of "style loss". The content loss is a measure of the difference between features in a given layer when the content image and current generated
image are passed into the CNN. We also use a "TV loss" to encourage smoothness in the image by penalizing variations between nearby pixels. The summation
of these 3 losses produces a quanity we can optimize the image over (through backprop into the image features) to produce an image that captures 
the syle of the style image and the content of the content image.
