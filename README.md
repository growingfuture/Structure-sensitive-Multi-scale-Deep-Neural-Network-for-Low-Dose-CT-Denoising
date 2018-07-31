# Structure-sensitive-Multi-scale-Deep-Neural-Network-for-Low-Dose-CT-Denoising

#### Chenyu You, Qingsong Yang, Hongming Shan, Lars Gjesteby, Guang Li, Shenghong Ju, Zhuiyang Zhang, Zhen Zhao, Yi Zhang, Member, IEEE, Wenxiang Cong, and Ge Wang*, Fellow, IEEE

## Introduction


### Previous Research for Low Dose CT Denoising

* Numerous methods have been designed for noise reduction in LDCT. These methods can be categorized as follows:

1. Sinogram filtering-based techniques : these methods directly process projection data in the projection domain, bowtie filtering, and structural adaptive filtering. The
main advantage of these methods is computational efficiency. 
(bold) However, they may result in loss of structural information and spatial resolution in LDCT acquisition [6]–[8]

2. Iterative reconstruction(IR) : IR techniques may potentially produce high signal-to-noise ratio (SNR). 
However, these methods require substantial computational resource, optimal parameter settings and accurate modeling of noise properties; 

3. Image space denoising techniques : these techniques can be performed directly on reconstructed images so that they can be applied across various CT scanners at a very low
cost. Examples are non-local means-based filters, dictionary-learning-based K-singular value decomposition (KSVD) method [18] and the block-matching 3D (BM3D) algorithms [22],
[23]. Even though these algorithms greatly suppress noise and artifacts, edge blurring or loss of spatial resolution may still remain in the processed LDCT images.

* Deep Learning : 
Recent studies demonstrate that deep learning (DL) techniques have yielded successful results for noise reduction in LDCT.

1. Chen et al. [31] proposed a pioneering **Residual
Encoder-Decoder convolutional neural network (REN-CNN)**
to predict NDCT images from noisy LDCT images. This
method greatly reduces the background noise and artifacts.
*However, the major limitation is that the content of the
results is blurry since the method is iteratively minimizing
the mean-squared error per voxel between generated LDCT
and the corresponding NDCT images.*

2. To cope with this limitation,
**generative adversarial networks (GANs)** [36] provide a
feasible solution. The generator G learns to capture a real
data distribution Pr and the discriminator D attempts to
discriminate between the synthetic data distribution and the
real data distribution. Note that the loss used in GAN, called
adversarial loss, measures the distance between the synthetic
data distribution and the real data distribution in order to
improve the performance of G and D. Here the GAN uses
arXiv:1805.00587v2 [cs.CV] 4 May 2018 2 Jensen-Shannon (JS) divergence to evaluate the similarity of
the two data distributions [36]. *However, several problems
still exist in training GAN, such as unstable training or nonconvergence
issues.*

3. To cope with these issues, Arjovsky et al.
introduced the **Wasserstein distance** instead of Jensen-Shannon
divergence to improve the stability of the neural network
training [37]. We discuss more details in Section II-D3.

4. Our previous work [33], which first introduced perceptual
loss to capture perceptual differences between denoised
LDCT images and the reference NDCT images, provides the
perceptually better results for clinical diagnosis *but yields
low scores in image quality metrics.* 



###  Contributions & innovations

1. *To better render the underlying structural information
between LDCT and NDCT images*, we adopt a **3D CNN
model as a generator** based on **WGAN** which can integrate
spatial information to enhance the image quality
and yield 3D volumetric results for better diagnosis.

2. *To consider the structural and perceptual difference
between generated LDCT images and gold-standard*,
**structure-sensitive loss** can enhance the accuracy and
robustness of the algorithm. Different from [33], we
replace perceptual loss with the **combination of L1 loss
and structural loss** to capture local anatomical structures
while reducing background noise.

3. *To better compare the performance of the 2D and the
3D models*, we perform **extensive investigations and
evaluations** on their convergence rate and denoising
performance.

## Methods

### Overview


### Network Structure


![Alt text](https://user-images.githubusercontent.com/37169177/43446438-d96fc016-94e3-11e8-83b9-ef9c102169f7.PNG)

#### 1) 3D CNN Generator

* The generator G consists of eight
3D convolutional (Conv) layers. The first 7 layers each have
32 filters and the last layer has only 1 filter. 

* Based on our
practical experience, the odd-numbered convolutional layers
have 3 * 3 * 1 filters, and the even-numbered convolutional
layers have 3 * 3 * 3 filters. The size of the extracted 3D
patches used as the input is 80 * 80 * 11. 

* See Fig. 1. Note
that the variable n denotes the number of the filters and
s denotes the stride size which means the step size of the
filer when moving across the image so that n32s1 stands
for 32 feature maps with stride one pixel. 

* A pooling layer after each Conv layer may lead to loss of subtle
textural and structural information, and spatial inconsistency
in training stages. Therefore, the pooling layer is not applied
in this network. 

* Next, the Rectified Linear Unit (ReLU) [41]
is utilized as the activation function after each Conv layer. The
benefits to utilize ReLu are as follows. First, it produces nonlinear
interactions with its input, which in turn prevents the
generated results from being equal to a linear transformation
of the input. Second, it is crucial to maintain the sparsity
in the inputs to each Conv layer to perform effective feature
extraction in high-dimensional feature space 

#### 2) Structure-Sensitive Loss (SSL) Function
The proposed 3D SSL function measures the patch-wise error between the
3D output from 3D ConvNet and the 3D NDCT images in spatial domain. This error was back-propagated [42] through
the neural network to update the weights of network. 

#### 3) Discriminator
The discriminator D used in this network is made up of six convolutional layers with 64, 64, 128,
128, 256, and 256 filters and the kernel size of 3 * 3. Two fully-connected (FC) layers produce 1024 and 1 feature maps
respectively. Each layer is followed by a leaky ReLU in the
manner of max(0; x) - *a* max(0;-x) [41] where *a* is a small
constant.

* A stride of one pixel is applied for odd-numbered
Conv layers and a stride of two pixels for even-numbered Conv
layers. The input fed to D has the size of 64 * 64 * 3, and is
the output of G. The reason why we use a 2D filter in D is
to lower the computational complexity. Since the adversarial
loss between each two adjacent layers in one volumetric patch
contribute equally to the weighted average in one iteration, it
adds no computational expense and also takes into account
spatial correlations. Following the suggestion in [37], we
removes the sigmoid cross entropy layer in D.


### Loss Functions for Noise Reduction

(formula and explains)

## Experiments and Results
