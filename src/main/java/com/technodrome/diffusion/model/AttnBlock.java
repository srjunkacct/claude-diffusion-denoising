package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.technodrome.diffusion.nn.GroupNormBlock;
import com.technodrome.diffusion.nn.NinBlock;

/**
 * Self-attention block with Q/K/V via NIN (1x1 conv), reshape+matmul approach.
 * Ported from unet.py: attn_block().
 *
 * Input: NDList[x: [B, C, H, W]]
 * Output: NDList[result: [B, C, H, W]]
 *
 * Original uses einsum in NHWC; we use reshape+matmul in NCHW:
 *   q,k,v: [B,C,H,W] -> reshape [B,C,N] where N=H*W
 *   w = q^T @ k / sqrt(C) -> [B,N,N]
 *   h = w @ v^T -> [B,N,C] -> transpose -> [B,C,N] -> reshape [B,C,H,W]
 */
public class AttnBlock extends AbstractBlock {

    private final int channels;
    private final GroupNormBlock norm;
    private final NinBlock qProj;
    private final NinBlock kProj;
    private final NinBlock vProj;
    private final NinBlock projOut;

    public AttnBlock(int channels) {
        this.channels = channels;
        norm = addChildBlock("norm", new GroupNormBlock(32, channels));
        qProj = addChildBlock("q", new NinBlock(channels));
        kProj = addChildBlock("k", new NinBlock(channels));
        vProj = addChildBlock("v", new NinBlock(channels));
        projOut = addChildBlock("proj_out", new NinBlock(channels, 0.0f)); // init_scale=0
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        norm.initialize(manager, dataType, inputShapes);
        qProj.initialize(manager, dataType, inputShapes);
        kProj.initialize(manager, dataType, inputShapes);
        vProj.initialize(manager, dataType, inputShapes);
        projOut.initialize(manager, dataType, inputShapes);
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        Shape shape = x.getShape();
        long B = shape.get(0), C = shape.get(1), H = shape.get(2), W = shape.get(3);
        long N = H * W;

        // Normalize
        NDArray h = norm.forward(ps, new NDList(x), training).singletonOrThrow();

        // Q, K, V projections: [B, C, H, W]
        NDArray q = qProj.forward(ps, new NDList(h), training).singletonOrThrow();
        NDArray k = kProj.forward(ps, new NDList(h), training).singletonOrThrow();
        NDArray v = vProj.forward(ps, new NDList(h), training).singletonOrThrow();

        // Reshape to [B, C, N]
        q = q.reshape(B, C, N);
        k = k.reshape(B, C, N);
        v = v.reshape(B, C, N);

        // Attention weights: w = q^T @ k * scale -> [B, N, N]
        double scale = Math.pow(C, -0.5);
        NDArray w = q.transpose(0, 2, 1).matMul(k).mul(scale); // [B, N, C] @ [B, C, N] = [B, N, N]
        w = w.softmax(-1);

        // Apply attention to values: h = w @ v^T -> [B, N, C]
        h = w.matMul(v.transpose(0, 2, 1)); // [B, N, N] @ [B, N, C] = [B, N, C]

        // Transpose back and reshape: [B, N, C] -> [B, C, N] -> [B, C, H, W]
        h = h.transpose(0, 2, 1).reshape(B, C, H, W);

        // Output projection (init_scale=0)
        h = projOut.forward(ps, new NDList(h), training).singletonOrThrow();

        return new NDList(x.add(h));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }
}
