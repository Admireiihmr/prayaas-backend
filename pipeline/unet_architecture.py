"""U-Net architecture, shared between training (segmentation.ipynb) and
serving (model_loader.load_segmentation_model). Keeping one definition means
weights saved by one always match the graph built by the other -- unlike a
full-model .keras file, which can fail to deserialize across tf_keras minor
versions (see the ValueError this replaced: "Layer 'conv2d' expected 1
variables, but received 0 variables").
"""

IMG_SIZE = 256


def conv_block(x, filters):
    import tf_keras.layers as layers

    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(img_size=IMG_SIZE, filters=(32, 64, 128, 256, 512)):
    import tf_keras
    import tf_keras.layers as layers

    inputs = layers.Input((img_size, img_size, 3))

    skips = []
    x = inputs
    for f in filters[:-1]:
        x = conv_block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

    x = conv_block(x, filters[-1])

    for f, skip in zip(reversed(filters[:-1]), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf_keras.Model(inputs, outputs, name="unet")
