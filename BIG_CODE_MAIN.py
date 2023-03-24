import time
import numpy as np
import tensorflow as tf
import os
import subprocess
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import imageio
import imghdr
import png

def make_generator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512 * 512,input_shape=(100,),use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((64, 64, 64)))
    assert model.output_shape == (None, 64, 64, 64)

    model.add(tf.keras.layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',use_bias=False))
    assert model.output_shape == (None, 128, 128, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',use_bias=False))
    assert model.output_shape == (None, 256, 256, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',activation='tanh',use_bias=False))
    assert model.output_shape == (None, 512, 512, 3)

    return model

def make_discriminator_model():
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding='same'))
  model.add(tf.keras.layers.LeakyReLU())

  model.add(tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.2))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(1))

  return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output,fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output),real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output),fake_output)

images = []
Count = 0
for i in os.scandir('AIPICS/'):
    Count += 1
    img_type = imghdr.what(i.path)
    if img_type is None:
        print(f"{i.path} is not an image")
    else:
        images.append(i.path)
print(Count)
images = tf.data.Dataset.from_tensor_slices(images)

def get_ds(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img,channels=3)
  img = tf.image.convert_image_dtype(img,tf.float32)
  img = tf.divide(tf.subtract(tf.multiply(img,255),127.5),127.5)
  return tf.image.resize(img,(512,512))

BATCH_SIZE = 1
train_images = images.map(get_ds,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()
noise = tf.random.normal([1,100])
generated_image = generator(noise,training=False)

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

generator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)

EPOCHS = 1000
noise_dims = 100
num_egs_to_generate = 16
seed = tf.random.normal([num_egs_to_generate,noise_dims])
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE,noise_dims])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
    generated_images = generator(noise,training=True)

    real_output = discriminator(images,training=True)
    fake_output = discriminator(generated_images,training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output,fake_output)

  gen_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
  dis_gradients = dis_tape.gradient(disc_loss,discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gen_gradients,generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(dis_gradients,discriminator.trainable_variables))

def generate_and_save_output(model,epoch,test_input):

  predictions = model(test_input,training=False)
  print((np.asarray(predictions[0]) * 255).astype(np.uint8))
  print(f"image_{epoch}.png")
  im = PIL.Image.fromarray((np.asarray(predictions[0]) * 255).astype(np.uint8))
  im.save(f"image_{epoch}.png")

def train(dataset,epochs):
  for epoch in range(epochs):
    start = time.time()
    for batch in dataset:
      train_step(batch)
      #print(f'{time.time()-start}')
    generate_and_save_output(generator,epoch+1,seed)

    print(f'Time for epoch {epoch + 1} is {time.time()-start}')
  #generate_and_save_output(generator,epochs,seed)
train(train_images,EPOCHS)
