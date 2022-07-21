# define the generator model
model = Sequential()
# foundation for 8x8 image
n_nodes = 64 * 8 * 8
model.add(Dense(n_nodes, input_dim=100))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Reshape((8, 8, 64)))
# upsample to 16x16
model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
# upsample to 32x32
model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
# upsample to 64x64
model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
