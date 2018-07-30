# Structure-sensitive-Multi-scale-Deep-Neural-Network-for-Low-Dose-CT-Denoising

#### Chenyu You, Qingsong Yang, Hongming Shan, Lars Gjesteby, Guang Li, Shenghong Ju, Zhuiyang Zhang, Zhen Zhao, Yi Zhang, Member, IEEE, Wenxiang Cong, and Ge Wang*, Fellow, IEEE

## Introduction


### Previous Research for Low Dose CT Denoising

* Numerous methods have been designed for noise reduction in LDCT. These methods can be categorized as follows:
>
> 1. Sinogram filtering-based techniques [4]–[8]: these methods directly process projection data in the projection domain [6], bowtie filtering [7], and structural adaptive filtering [8]. The
main advantage of these methods is computational efficiency. 
(bold) However, they may result in loss of structural information and spatial resolution in LDCT acquisition [6]–[8]
>
> 2. Iterative reconstruction(IR) : IR techniques may potentially produce high signal-to-noise ratio (SNR). 
However, these methods require substantial computational resource, optimal parameter settings and accurate modeling of noise properties; 
>
> 3. Image space denoising techniques : these techniques can be performed directly on reconstructed images so that they can be applied across various CT scanners at a very low
cost. Examples are non-local means-based filters, dictionary-learning-based K-singular value decomposition (KSVD) method [18] and the block-matching 3D (BM3D) algorithms [22],
[23]. Even though these algorithms greatly suppress noise and artifacts, edge blurring or loss of spatial resolution may still remain in the processed LDCT images.

* Deep Learning : 
>Recent studies demonstrate that deep learning (DL) techniques have yielded successful results for noise reduction in LDCT.
>
> * Chen et al. [31] proposed a pioneering Residual
Encoder-Decoder convolutional neural network (REN-CNN)
to predict NDCT images from noisy LDCT images. This
method greatly reduces the background noise and artifacts.
**However, the major limitation is that the content of the
results is blurry since the method is iteratively minimizing
the mean-squared error per voxel between generated LDCT
and the corresponding NDCT images.**
>
* To cope with this limitation,
generative adversarial networks (GANs) [36] provide a
feasible solution. The generator G learns to capture a real
data distribution Pr and the discriminator D attempts to
discriminate between the synthetic data distribution and the
real data distribution. Note that the loss used in GAN, called
adversarial loss, measures the distance between the synthetic
data distribution and the real data distribution in order to
improve the performance of G and D. Here the GAN uses
arXiv:1805.00587v2 [cs.CV] 4 May 2018 2 Jensen-Shannon (JS) divergence to evaluate the similarity of
the two data distributions [36]. **However, several problems
still exist in training GAN, such as unstable training or nonconvergence
issues.**

* To cope with these issues, Arjovsky et al.
introduced the **Wasserstein distance** instead of Jensen-Shannon
divergence to improve the stability of the neural network
training [37]. We discuss more details in Section II-D3.

* Our previous work [33], which first introduced perceptual
loss to capture perceptual differences between denoised
LDCT images and the reference NDCT images, provides the
perceptually better results for clinical diagnosis **but yields
low scores in image quality metrics.** 



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



## Experiments and Results
