import cv2
import numpy as np
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast
)
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

EPOCHS = 150
IMG_SIZE = 256


BASE_PATH="/home/mcpdschaves/visaocomputacional"

os.makedirs(BASE_PATH+"/models/checkpoints",exist_ok=True)

os.makedirs(BASE_PATH+"/logs",exist_ok=True)

ROOT=BASE_PATH+"/dataset"

def augment(image, mask):

    # Flip horizontal
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Flip vertical
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Rotação 0°,90°,180°,270°
    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32
    )

    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Brilho
    image = tf.image.random_brightness(
        image,
        max_delta=0.15
    )

    # Contraste
    image = tf.image.random_contrast(
        image,
        lower=0.8,
        upper=1.2
    )

    return image, mask


def load_pair(
    image_path,
    mask_path
):


    image=tf.io.read_file(
        image_path
    )


    image=tf.image.decode_jpeg(
        image,
        channels=3
    )


    image=tf.image.resize(
        image,
        (IMG_SIZE,IMG_SIZE)
    )


    image=tf.cast(
        image,
        tf.float32
    )/255.0



    mask=tf.io.read_file(
        mask_path
    )


    mask=tf.image.decode_png(
        mask,
        channels=1
    )


    mask=tf.image.resize(
        mask,
        (IMG_SIZE,IMG_SIZE),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


    mask=tf.cast(
        mask>0,
        tf.float32
    )


    return image,mask

# Dataset Tensorflow
def get_pairs(img_dir, mask_dir):

    images = []

    masks = []

    for img in glob(f"{img_dir}/*.jpg"):

        name = os.path.splitext(
            os.path.basename(img)
        )[0]

        mask = os.path.join(
            mask_dir,
            name + ".png"
        )

        if os.path.exists(mask):

            images.append(img)
            masks.append(mask)

    return images, masks

train_images, train_masks = get_pairs(
    f"{ROOT}/train",
    f"{ROOT}/train/masks"
)


val_images, val_masks = get_pairs(
    f"{ROOT}/valid",
    f"{ROOT}/valid/masks"
)

#dataset de treino
def load_train(
    image,
    mask
):

    image,mask=load_pair(
        image,
        mask
    )

    return image,mask

assert len(train_images) > 0
assert len(train_masks) > 0
assert len(val_images) > 0
assert len(val_masks) > 0


train_dataset=tf.data.Dataset.from_tensor_slices(
    (
        train_images,
        train_masks
    )
)


train_dataset=train_dataset.map(
    load_train,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_dataset=train_dataset.batch(
    8
)


train_dataset=train_dataset.prefetch(
    tf.data.AUTOTUNE
)

# dataset validação
val_dataset=tf.data.Dataset.from_tensor_slices(
    (
        val_images,
        val_masks
    )
)


val_dataset=val_dataset.map(
    load_pair,
    num_parallel_calls=tf.data.AUTOTUNE
)


val_dataset=val_dataset.batch(
    8
)


val_dataset=val_dataset.prefetch(
    tf.data.AUTOTUNE
)

#Rede U-net

def conv_block(
    x,
    filters
):

    x=Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x=BatchNormalization()(x)

    x=Activation(
        "relu"
    )(x)



    x=Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x=BatchNormalization()(x)

    x=Activation(
        "relu"
    )(x)


    return x




def build_unet():


    inputs=Input(
        (256,256,3)
    )


    c1=conv_block(inputs,32)
    p1=MaxPooling2D()(c1)


    c2=conv_block(p1,64)
    p2=MaxPooling2D()(c2)


    c3=conv_block(p2,128)
    p3=MaxPooling2D()(c3)


    c4=conv_block(p3,256)
    p4=MaxPooling2D()(c4)


    bn=conv_block(p4,512)



    u1=UpSampling2D()(bn)

    u1=concatenate(
        [u1,c4]
    )

    c5=conv_block(
        u1,
        256
    )



    u2=UpSampling2D()(c5)

    u2=concatenate(
        [u2,c3]
    )

    c6=conv_block(
        u2,
        128
    )



    u3=UpSampling2D()(c6)

    u3=concatenate(
        [u3,c2]
    )

    c7=conv_block(
        u3,
        64
    )



    u4=UpSampling2D()(c7)

    u4=concatenate(
        [u4,c1]
    )


    c8=conv_block(
        u4,
        32
    )



    outputs=Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c8)



    return Model(
        inputs,
        outputs
    )

#Loss e métricas
def dice_loss(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    smooth = 1e-7

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )

    return 1.0 - dice



def combined_loss(
    y_true,
    y_pred
):

    bce=tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )


    return bce+dice_loss(
        y_true,
        y_pred
    )

def dice_metric(y_true, y_pred):

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2.0 * intersection + 1e-7
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + 1e-7
    )



def iou_metric(y_true, y_pred):

    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

#Compilar
model=build_unet()



model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss=combined_loss,

    metrics=[
        dice_metric,
        iou_metric
    ]

)


model.summary()

#Callback para salvar progresso
from tensorflow.keras.callbacks import *



checkpoint=ModelCheckpoint(

    BASE_PATH+
    "/models/checkpoints/unet_best.keras",

    monitor="val_dice_metric",

    save_best_only=True,

    mode="max",

    verbose=1

)



early=EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True

)



reduce=ReduceLROnPlateau(

    monitor="val_loss",

    patience=5,

    factor=0.5

)

print("Imagens treino:",len(train_images))
print("Máscaras treino:",len(train_masks))

print("Imagens validação:",len(val_images))
print("Máscaras validação:",len(val_masks))

import tensorflow as tf

print(tf.config.list_physical_devices())

print(
"Proporção:",
len(train_images)==len(train_masks)
)

for image, mask in train_dataset.take(1):

    print("Imagem:")
    print(image.shape)

    print("Mascara:")
    print(mask.shape)

img=cv2.imread(train_images[0])

img=cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


mask=cv2.imread(
    train_masks[0],
    0
)


plt.figure(figsize=(10,5))


plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Imagem")


plt.subplot(1,2,2)
plt.imshow(mask,cmap="gray")
plt.title("Máscara")

plt.savefig(BASE_PATH+"/sample_mask.png")
plt.close()

mask_test = cv2.imread(
    train_masks[0],
    0
)

print(
    "Valores únicos máscara:",
    np.unique(mask_test)
)

print(
    "Pixels positivos:",
    np.sum(mask_test > 0)
)

#treinamento
history=model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early,
        reduce
    ]

)

model.save(
BASE_PATH+"/models/unet_final.keras"
)

# salvar histórico
import json


with open(
BASE_PATH+"/logs/history.json",
"w"
) as f:

    json.dump(
        history.history,
        f
    )

GRAPH_PATH = BASE_PATH + "/logs/graphs"


os.makedirs(
    GRAPH_PATH,
    exist_ok=True
)

plt.figure(figsize=(10,6))


plt.plot(
    history.history["loss"],
    label="Treino"
)


plt.plot(
    history.history["val_loss"],
    label="Validação"
)


plt.xlabel("Época")

plt.ylabel("Loss")


plt.title(
    "Curva de perda"
)


plt.legend()


plt.grid()


plt.savefig(
    GRAPH_PATH+"/loss.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()

plt.figure(figsize=(10,6))


plt.plot(
    history.history["dice_metric"],
    label="Treino"
)


plt.plot(
    history.history["val_dice_metric"],
    label="Validação"
)


plt.xlabel("Época")

plt.ylabel("Dice")


plt.title(
    "Dice Score"
)


plt.legend()


plt.grid()


plt.savefig(
    GRAPH_PATH+"/dice.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()

plt.figure(figsize=(10,6))


plt.plot(
    history.history["iou_metric"],
    label="Treino"
)


plt.plot(
    history.history["val_iou_metric"],
    label="Validação"
)


plt.xlabel("Época")

plt.ylabel("IoU")


plt.title(
    "Intersection over Union"
)


plt.legend()


plt.grid()


plt.savefig(
    GRAPH_PATH+"/iou.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()

