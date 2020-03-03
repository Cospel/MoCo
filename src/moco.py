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
EPOCHS = 25
SAMPLES = 5000


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Replace BatchNormalization layers with this new layer.
    This layer has fixed momentum 0.9 so when we are doing
    transfer learning on small dataset the learning is a bit faster.

    Usage:
        tf.keras.layers.BatchNormalization = BatchNormalization

        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", input_shape=self.shape, include_top=False, layers=tf.keras.layers
        )
    """

    def __init__(self, momentum=0.9, name=None, **kwargs):
        super(BatchNormalization, self).__init__(momentum=0.9, name=name, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=training)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config


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
    return loss, k, logits,


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
    tf.keras.layers.BatchNormalization = BatchNormalization

    inputs = tf.keras.Input(shape=input_shape, name="input")
    model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top = False, weights=None, layers=tf.keras.layers)(inputs)
    pool = tf.keras.layers.GlobalAveragePooling2D()(model)
    results = tf.keras.layers.Dense(units = EMBEDDING_DIM, name="embeddings")(pool)
    results = tf.keras.layers.Lambda(lambda  x: tf.keras.backend.l2_normalize(x,axis=1))(results)
    return tf.keras.Model(inputs = inputs, outputs = results)


def Predictor(base_model):
    outputs = tf.keras.layers.Activation("relu")(base_model.get_layer("embeddings").output)
    outputs = tf.keras.layers.Dense(units = CLASSES, activation="softmax")(outputs)
    return tf.keras.Model(inputs = base_model.input, outputs = outputs)


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


def parse_function_moco(image, label):
    data = tf.cast(image, dtype = tf.float32)
    x = (data / 127.5) - 1
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x_aug_1 = augment(x)
    x_aug_2 = augment(x)
    return x_aug_1, x_aug_2, label


def parse_function_supervised(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = augment(image)
    return image, tf.reduce_sum(tf.one_hot([label], CLASSES), axis=0)


# Load dataset
trainset, testset = tfds.load(
            name = 'deep_weeds', #'colorectal_histology', 'deep_weeds'
            split=['train[:80%]',  'train[80%:]'],
            as_supervised=True)

trainset_s = trainset.map(parse_function_supervised).batch(BATCH_SIZE)
testset_s = testset.map(parse_function_supervised).batch(BATCH_SIZE)
trainset_moco = iter(trainset.map(parse_function_moco).repeat(EPOCHS).shuffle(SAMPLES, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(200))

# Create an instance of the model
model_query = Encoder(input_shape)
model_keys = Encoder(input_shape)

# Initialise the models and make the EMA model 90% similar to the main model
update_model_via_ema(model_query, model_keys, 0.1)

# init optimizer
STEPS = len(list(trainset_s))

lr = tf.keras.optimizers.schedules.PolynomialDecay(
    0.00001, STEPS, end_learning_rate=0.02, power=1.0,
    cycle=True, name=None
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

# Initialize queue with some size
queue = MoCoQueue(EMBEDDING_DIM, 512)

# Initialize loss metrics watcher
pred_meter_pos = AverageMeter()
pred_meter_neg = AverageMeter()
loss_meter = AverageMeter()

for epoch in range(EPOCHS):
    with tqdm(total=STEPS, ncols=200) as pbar:
        i = 0
        for x1, x2, y in trainset_moco:
            start = time.time()
            loss = moco_training_step(x1, x2, queue, model_query, model_keys, optimizer, pred_meter_pos, pred_meter_neg, loss_meter)

            pbar.set_postfix(loss=loss.numpy(), epoch=epoch, lr=optimizer.learning_rate(optimizer.iterations).numpy(), neg=pred_meter_neg.avg, pos=pred_meter_pos.avg, loss_avg=loss_meter.avg, step_time=time.time()-start)
            pbar.update(1)

            if i == (STEPS - 1):
                break

            i += 1

    loss_meter.reset()
    pred_meter_pos.reset()
    pred_meter_neg.reset()

# train supervised
model_p = Predictor(model_keys)
model_p.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

model_p.fit(trainset_s, validation_data=testset_s, epochs=EPOCHS)
