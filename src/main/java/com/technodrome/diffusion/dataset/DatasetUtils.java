package com.technodrome.diffusion.dataset;

import ai.djl.ndarray.NDArray;

/**
 * Dataset utility functions for preprocessing.
 * Ported from tpu_utils.py: normalize_data(), unnormalize_data().
 */
public final class DatasetUtils {

    private DatasetUtils() {}

    /**
     * Normalize image from [0, 255] uint8 to [-1, 1] float32.
     */
    public static NDArray normalize(NDArray x) {
        return x.toType(ai.djl.ndarray.types.DataType.FLOAT32, false)
                .div(127.5f)
                .sub(1.0f);
    }

    /**
     * Unnormalize image from [-1, 1] float32 to [0, 255] uint8.
     */
    public static NDArray unnormalize(NDArray x) {
        return x.add(1.0f)
                .mul(127.5f)
                .clip(0, 255)
                .toType(ai.djl.ndarray.types.DataType.UINT8, false);
    }

    /**
     * Unnormalize and convert NCHW float32 [-1,1] to NHWC uint8 [0,255].
     * Suitable for image saving.
     */
    public static NDArray toImageBatch(NDArray x) {
        // NCHW -> NHWC
        NDArray nhwc = x.transpose(0, 2, 3, 1);
        return unnormalize(nhwc);
    }
}
