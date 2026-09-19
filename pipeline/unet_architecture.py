"""U-Net architecture, shared between training (segmentation.ipynb) and
serving (model_loader.load_segmentation_model). Keeping one definition means
weights saved by one always match the graph built by the other -- unlike a
full-model .keras file, which can fail to deserialize across tf_keras minor
versions (see the ValueError this replaced: "Layer 'conv2d' expected 1
variables, but received 0 variables").

Encoder is a frozen, ImageNet-pretrained MobileNetV2, not a from-scratch
stack: with only ~524 training images, a from-scratch encoder has too little
data to learn useful low-level features, which capped foreground IoU at
~0.48. Transfer learning is the standard fix for segmentation with this
little labelled data.
"""

IMG_SIZE = 256

# MobileNetV2 layer names at each downsampling stage, for skip connections.
# See https://www.tensorflow.org/tutorials/images/segmentation -- same
# layers, adapted to our 256x256 input (stride 2/4/8/16/32 respectively).
_SKIP_LAYER_NAMES = [
    "block_1_expand_relu",   # 128x128
    "block_3_expand_relu",   # 64x64
    "block_6_expand_relu",   # 32x32
    "block_13_expand_relu",  # 16x16
]
_BOTTLENECK_LAYER_NAME = "block_16_project"  # 8x8


def _decoder_block(x, skip, filters):
    import tf_keras.layers as layers

    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(img_size=IMG_SIZE, pretrained=True):
    """pretrained=True downloads ImageNet weights for the encoder -- only
    needed when training a fresh model. Serving code (model_loader.py)
    passes pretrained=False and immediately overwrites every weight via
    load_weights(), so there's no reason to hit the network on every cold
    start for weights that are about to be discarded anyway.
    """
    import tf_keras
    import tf_keras.layers as layers

    base = tf_keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base.trainable = False  # frozen: ~524 images is too little to fine-tune 2.2M+ params safely

    skip_outputs = [base.get_layer(name).output for name in _SKIP_LAYER_NAMES]
    bottleneck = base.get_layer(_BOTTLENECK_LAYER_NAME).output
    encoder = tf_keras.Model(inputs=base.input, outputs=[*skip_outputs, bottleneck], name="mobilenetv2_encoder")
    encoder.trainable = False

    inputs = layers.Input((img_size, img_size, 3))
    # Our pipeline feeds [0, 1]-scaled RGB; MobileNetV2 expects [-1, 1]
    # (what mobilenet_v2.preprocess_input does starting from [0, 255]).
    x = layers.Lambda(lambda t: t * 2.0 - 1.0, name="rescale_to_imagenet_range")(inputs)
    *skips, x = encoder(x)

    for filters, skip in zip((256, 128, 64, 32), reversed(skips)):
        x = _decoder_block(x, skip, filters)

    # One more upsample (128x128 -> 256x256): no encoder skip exists at full
    # resolution since MobileNetV2's first feature map is already stride 2.
    x = _decoder_block(x, None, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf_keras.Model(inputs, outputs, name="unet_mobilenetv2")
