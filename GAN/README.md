# GAN

### 3 Uses for GANs

1. Fake faces.
2. Image deblurring.
3. 3D object generation from 2D image.


### GAN Hints

1. Downsample Using Strided Convolutions (e.g. don’t use pooling layers).
2. Upsample Using Strided Convolutions (e.g. use the transpose convolutional layer).
3. Use LeakyReLU (e.g. don’t use the standard ReLU).
4. Use BatchNormalization (e.g. standardize layer outputs after the activation).
5. Use Gaussian Weight Initialization (e.g. a mean of 0.0 and stdev of 0.02).
6. Use Adam Stochastic Gradient Descent (e.g. learning rate of 0.0002 and beta1 of 0.5).
7. Scale Images to the Range [-1,1] (e.g. use Tanh in the output of the generator).

#### For more GAN Hints

https://github.com/soumith/ganhacks


