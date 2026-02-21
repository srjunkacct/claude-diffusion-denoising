package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

/**
 * Sinusoidal timestep embedding from Fairseq.
 * Ported from nn.py: get_timestep_embedding().
 * Static utility — no learnable parameters.
 */
public final class TimestepEmbedding {

    private TimestepEmbedding() {}

    /**
     * Build sinusoidal embeddings for diffusion timesteps.
     *
     * @param timesteps    1D integer tensor of shape [B]
     * @param embeddingDim embedding dimension
     * @return tensor of shape [B, embeddingDim]
     */
    public static NDArray getTimestepEmbedding(NDArray timesteps, int embeddingDim) {
        NDManager manager = timesteps.getManager();
        int halfDim = embeddingDim / 2;

        double logFactor = Math.log(10000.0) / (halfDim - 1);
        // exp(-i * log(10000) / (halfDim - 1)) for i in [0, halfDim)
        NDArray freqs = manager.arange(halfDim).toType(DataType.FLOAT32, false).mul(-logFactor).exp();

        // timesteps[:, None] * freqs[None, :]
        NDArray args = timesteps.toType(DataType.FLOAT32, false).reshape(-1, 1)
                .mul(freqs.reshape(1, -1));

        // Concat [sin, cos]
        NDArray emb = NDArrays.concat(new NDList(args.sin(), args.cos()), 1);

        // Zero-pad if odd embedding dim
        if (embeddingDim % 2 == 1) {
            NDArray pad = manager.zeros(new ai.djl.ndarray.types.Shape(timesteps.size(), 1));
            emb = NDArrays.concat(new NDList(emb, pad), 1);
        }

        return emb; // [B, embeddingDim]
    }
}
