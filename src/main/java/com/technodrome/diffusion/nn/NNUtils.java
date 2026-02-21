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
        int ndim = x.getShape().dimension();
        int[] axes = new int[ndim - 1];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i + 1;
        }
        return x.sum(axes);
    }

    /** Mean over all dimensions except batch (dim 0). */
    public static NDArray meanflat(NDArray x) {
        int ndim = x.getShape().dimension();
        int[] axes = new int[ndim - 1];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i + 1;
        }
        return x.mean(axes);
    }
}
