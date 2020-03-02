"""
Contains various bits and pieces needed for the method described in the paper: 
Momentum Contrast for Unsupervised Visual Representation Learning 
(https://arxiv.org/abs/1911.05722)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow_datasets as tfds
from tqdm import tqdm

EMBEDDING_DIM = 64
CLASSES = 8
IMG_SIZE = 150
input_shape = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 20
EPOCHS = 2


def RandomAugmentation(input_shape, rotation_range = (-20, 20), scale_range = (0.8, 1.2), padding = 10, bri = 32./255., sat = (0.5, 1.5), hue = .2, con = (0.5, 1.5)):
    inputs = tf.keras.Input(input_shape[-3:])
    #results = RandomAffine(inputs.shape[-3:], rotation_range = rotation_range, scale_range = scale_range)(inputs)§
    results = tf.keras.layers.Lambda(lambda x, p: tf.image.resize(x, [tf.shape(x)[-3] + p, tf.shape(x)[-2] + p], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR), arguments = {'p': padding})(inputs)
    results = tf.keras.layers.Lambda(lambda x: tf.image.random_crop(x[0], size = tf.shape(x[1])))([results, inputs])
    results = tf.keras.layers.Lambda(lambda x: tf.image.random_flip_left_right(x))(results)
    results = tf.keras.layers.Lambda(lambda x: tf.image.random_flip_up_down(x))(results)
    results = tf.keras.layers.Lambda(lambda x, b: tf.image.random_brightness(x, b), arguments = {'b': bri})(results)
    results = tf.keras.layers.Lambda(lambda x, a, b: tf.image.random_saturation(x, lower = a, upper = b), arguments = {'a': sat[0], 'b': sat[1]})(results)
    results = tf.keras.layers.Lambda(lambda x, h: tf.image.random_hue(x, h), arguments = {'h': hue})(results)
    results = tf.keras.layers.Lambda(lambda x, a, b: tf.image.random_contrast(x, lower = a, upper = b), arguments = {'a': con[0], 'b': con[1]})(results)
    results = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0., 1.))(results)
    results = tf.keras.layers.Lambda(lambda x: tf.math.multiply(tf.math.subtract(x, 0.5), 2.))(results)
    return tf.keras.Model(inputs = inputs, outputs = results)


class MoCoQueue:
    def __init__(self, embedding_dim, max_queue_length):
        self.embedding_dim = embedding_dim
        # Put a single zeros key in there to start with, it will be pushed out eventually
        with tf.device("CPU:0"):
            self.keys = tf.random.normal([2, self.embedding_dim])
        self.max_queue_length = max_queue_length

    def enqueue(self, new_keys):
        self.keys = tf.concat([new_keys, self.keys], 0)
        if self.keys.shape[0] > self.max_queue_length:
            self.keys = self.keys[:self.max_queue_length]


#@tf.function
def _moco_training_step_inner(x, x_aug, queue, model_query, model_keys, temperature, optimizer):
    N = tf.shape(x)[0]
    K = tf.shape(queue)[0]
    C = tf.shape(queue)[1]
    k = model_keys(x_aug, training=True)  # no gradient
    with tf.GradientTape() as tape:
        q = model_query(x, training=True)
        l_pos = tf.matmul(tf.reshape(q, [N, 1, C]), tf.reshape(k, [N, C, 1]))
        l_pos = tf.reshape(l_pos, [N, 1])
        l_neg = tf.matmul(tf.reshape(q, [N, C]), tf.reshape(queue, [C, K]))
        logits = tf.concat([l_pos, l_neg], axis=1)
        # print(l_pos.numpy())
        # print(l_neg.numpy(), l_neg.numpy().shape)
        # print('-------')
        labels = tf.zeros([N], dtype="int64")
        loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(labels, logits / temperature)
        )
    gradients = tape.gradient(loss, model_query.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_query.trainable_variables))
    return loss, k


def moco_training_step(
    x,
    x_aug,
    queue,
    model_query,
    model_keys,
    optimizer,
    temperature=0.07,
    momentum=0.999,
):
    loss, new_keys = _moco_training_step_inner(
        x, x_aug, queue.keys, model_query, model_keys,
        tf.constant(temperature, dtype='float32'),
        optimizer
    )

    # update the EMA of the model
    update_model_via_ema(model_query, model_keys, momentum)
    queue.enqueue(new_keys)
    return loss


def update_model_via_ema(
    model_query, model_keys, momentum, just_trainable_vars=False
):
    iterable = (
        zip(model_query.trainable_variables, model_keys.trainable_variables)
        if just_trainable_vars
        else zip(model_query.variables, model_keys.variables)
    )
    for p, p2 in iterable:
        p2.assign(momentum * p2 + (1.0 - momentum) * p)


def Encoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name="input")
    model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top = False, weights=None)(inputs)
    pool = tf.keras.layers.GlobalAveragePooling2D()(model)
    results = tf.keras.layers.Dense(units = EMBEDDING_DIM, name="embeddings")(pool)
    results = tf.keras.layers.Lambda(lambda  x: tf.keras.backend.l2_normalize(x,axis=1))(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def Predictor(base_model):
    ouputs = tf.keras.layers.Dense(units = CLASSES, activation="softmax")(base_model.get_layer("embeddings").output)
    return tf.keras.Model(inputs = base_model.input, outputs = ouputs)

def parse_function(feature):
    data = (tf.cast(feature["image"], dtype = tf.float32) / 127.5) - 1
    label = feature["label"]
    return data, label

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, tf.reduce_sum(tf.one_hot([label], CLASSES), axis=0)


# Load dataset
trainset = tfds.load(
            name = 'colorectal_histology',
            as_supervised=True)

trainset_moco = iter(tfds.load(
            name = 'colorectal_histology',
            split = tfds.Split.TRAIN,
            download = False).repeat(-1).map(parse_function).shuffle(BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))

# Create an instance of the model
model_query = Encoder(input_shape)
model_keys = Encoder(input_shape)

# Initialise the models and make the EMA model 90% similar to the main model
update_model_via_ema(model_query, model_keys, 0.1)

queue = MoCoQueue(EMBEDDING_DIM, 256)
optimizer = tf.keras.optimizers.Adam()
train = trainset['train'].map(format_example).batch(BATCH_SIZE)

augmentor = RandomAugmentation(input_shape)

for epoch in range(EPOCHS):
    batch_x, batch_x_aug = [], []
    i = 0
    k = 0
    with tqdm(total=5000/BATCH_SIZE, ncols=100) as pbar:
        for x, y in trainset_moco:
            if i < BATCH_SIZE:
                #x_aug = x + 0.1 * tf.random.normal(x.shape, dtype='float32')
                x_aug = augmentor(x)
                x = augmentor(x)
                batch_x.append(x)
                batch_x_aug.append(x_aug)
                i += 1

            if i == BATCH_SIZE:
                #print('step')
                loss = moco_training_step(batch_x, batch_x_aug, queue, model_query, model_keys, optimizer)
                pbar.set_postfix(loss=loss.numpy())
                i = 0
                batch_x, batch_x_aug = [], []
                pbar.update(2)
            k += 1
            if k == 2000:
                break

# train supervised
model_p = Predictor(model_keys)
model_p.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

model_p.fit(train, epochs=EPOCHS)