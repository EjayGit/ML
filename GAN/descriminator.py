# define the discriminator model
model = Sequential()
# downsample to 32x32
model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(64,64,3)))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
# downsample to 16x16
model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
# downsample to 8x8
model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
# classify
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
