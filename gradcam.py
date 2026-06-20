import tensorflow as tf
import numpy as np
import matplotlib.cm as cm

keras_model = tf.keras.models.load_model(
    "brain_tumor_model.keras"
)

def make_gradcam_heatmap(
    img_array,
    pred_index=None,
    last_conv_layer_name="top_conv"
):

    grad_model = tf.keras.models.Model(
        keras_model.inputs,
        [
            keras_model.get_layer(
                last_conv_layer_name
            ).output,
            keras_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[
            :, pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[
        ..., tf.newaxis
    ]

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    ) / tf.math.reduce_max(
        heatmap
    )

    return heatmap.numpy()


def overlay_gradcam(
    original_image,
    heatmap,
    alpha=0.4
):

    heatmap = np.uint8(
        255 * heatmap
    )

    jet = cm.get_cmap(
        "jet"
    )

    jet_colors = jet(
        np.arange(256)
    )[:, :3]

    jet_heatmap = jet_colors[
        heatmap
    ]

    jet_heatmap = tf.keras.utils.array_to_img(
        jet_heatmap
    )

    jet_heatmap = jet_heatmap.resize(
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    jet_heatmap = tf.keras.utils.img_to_array(
        jet_heatmap
    )

    superimposed_img = (
        jet_heatmap * alpha
    ) + original_image

    return np.uint8(
        superimposed_img
    )
