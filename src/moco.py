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
import time
from tensorflow.python.keras import backend as K

EMBEDDING_DIM = 64
CLASSES = 8
IMG_SIZE = 150
input_shape = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 100
EPOCHS = 50
SAMPLES = 5000


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


@tf.function(experimental_relax_shapes=True)
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
        labels = tf.zeros([N], dtype="int64")
        loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(labels, logits / temperature)
        )

    gradients = tape.gradient(loss, model_query.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_query.trainable_variables))
    return loss, k, logits


def moco_training_step(
    x,
    x_aug,
    queue,
    model_query,
    model_keys,
    optimizer,
    pred_meter_pos, pred_meter_neg, loss_meter,
    temperature=0.07,
    momentum=0.999,
):
    loss, new_keys, logits = _moco_training_step_inner(
        x, x_aug, queue.keys, model_query, model_keys,
        tf.constant(temperature, dtype='float32'),
        optimizer
    )

    # update some stats
    pred_meter_pos.update(logits.numpy()[:, 0].mean())
    pred_meter_neg.update(logits.numpy()[:, 1:].mean())
    loss_meter.update(loss.numpy())

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

@tf.function
def augment(image, padding = 10, bri = 32./255., sat = (0.5, 1.5), hue = .2, con = (0.5, 1.5)):
    x = image
    x = tf.image.resize(x, [tf.shape(x)[-3] + padding, tf.shape(x)[-2] + padding], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.image.random_crop(x, size = tf.shape(image))
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, bri)
    x = tf.image.random_saturation(x, lower = sat[0], upper = sat[1])
    x = tf.image.random_hue(x, hue)
    x = tf.image.random_contrast(x, lower = con[0], upper = con[1])
    #x = tf.clip_by_value(x, -1., 1.)
    x = tf.math.multiply(tf.math.subtract(x, 0.5), 2.)
    return x

def parse_function(image, label):
    data = tf.cast(image, dtype = tf.float32)
    x = (data / 127.5) - 1
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x_aug_1 = augment(x)
    x_aug_2 = augment(x)
    return x_aug_1, x_aug_2, label

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, tf.reduce_sum(tf.one_hot([label], CLASSES), axis=0)

# Load dataset
trainset = tfds.load(
            name = 'colorectal_histology',
            as_supervised=True)['train'].map(format_example).batch(BATCH_SIZE)

trainset_moco = iter(tfds.load(
            name = 'colorectal_histology',
            as_supervised=True)['train'].map(parse_function).repeat(EPOCHS).shuffle(SAMPLES, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(200))

# Create an instance of the model
model_query = Encoder(input_shape)
model_keys = Encoder(input_shape)

# Initialise the models and make the EMA model 90% similar to the main model
update_model_via_ema(model_query, model_keys, 0.1)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, nesterov=True)# weight_decay=1e-5)

queue = MoCoQueue(EMBEDDING_DIM, 1024)

pred_meter_pos = AverageMeter()
pred_meter_neg = AverageMeter()
loss_meter = AverageMeter()

for epoch in range(EPOCHS):
    with tqdm(total=SAMPLES/BATCH_SIZE, ncols=200) as pbar:
        i = 0
        for x1, x2, y in trainset_moco:
            start = time.time()
            loss = moco_training_step(x1, x2, queue, model_query, model_keys, optimizer, pred_meter_pos, pred_meter_neg, loss_meter)
            if i == ((SAMPLES/BATCH_SIZE) - 1):
                break

            pbar.set_postfix(loss=loss.numpy(), epoch=epoch, neg=pred_meter_neg.avg, pos=pred_meter_pos.avg, loss_avg=loss_meter.avg, step_time=time.time()-start)
            pbar.update(1)
            i += 1

    if epoch == EPOCHS/2:
        print("Changing learning rate....!")
        K.set_value(optimizer.lr, 0.002)

    loss_meter.reset()
    pred_meter_pos.reset()
    pred_meter_neg.reset()

# train supervised
model_p = Predictor(model_keys)
model_p.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

model_p.fit(trainset, epochs=EPOCHS)
