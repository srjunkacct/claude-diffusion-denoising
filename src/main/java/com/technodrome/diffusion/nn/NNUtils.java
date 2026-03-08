package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDArray;

/**
 * Static utility methods ported from nn.py: sumflat, meanflat, flatten.
 */
public final class NNUtils {

    private NNUtils() {}

    /** Flatten tensor to 2D: [B, ...] -> [B, -1]. */
    public static NDArray flatten(NDArray x) {
        long batch = x.getShape().get(0);
        return x.reshape(batch, -1);
    }

    /** Sum over all dimensions except batch (dim 0). */
    public static NDArray sumflat(NDArray x) {
        // DJL PyTorch only supports single-axis reduction; flatten then sum
        return flatten(x).sum(new int[]{1});
    }

    /** Mean over all dimensions except batch (dim 0). */
    public static NDArray meanflat(NDArray x) {
        // DJL PyTorch only supports single-axis reduction; flatten then mean
        return flatten(x).mean(new int[]{1});
    }
}
