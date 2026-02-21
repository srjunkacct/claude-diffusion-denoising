package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.technodrome.diffusion.nn.Conv2dBlock;

/**
 * 2x spatial upsampling via nearest-neighbor interpolation + optional 3x3 conv.
 * Ported from unet.py: upsample().
 * Input/output: NCHW [B, C, H, W] -> [B, C, 2H, 2W].
 */
public class UpsampleBlock extends AbstractBlock {

    private final boolean withConv;
    private Conv2dBlock conv;

    public UpsampleBlock(int channels, boolean withConv) {
        this.withConv = withConv;
        if (withConv) {
            conv = addChildBlock("conv", new Conv2dBlock(channels, 3, 1, 1.0f, true));
        }
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        Shape shape = x.getShape();
        long B = shape.get(0), C = shape.get(1), H = shape.get(2), W = shape.get(3);

        // Nearest-neighbor 2x upsample: reshape + tile
        x = x.reshape(B, C, H, 1, W, 1)
             .tile(new long[]{1, 1, 1, 2, 1, 2})
             .reshape(B, C, H * 2, W * 2);

        if (withConv) {
            x = conv.forward(ps, new NDList(x), training).singletonOrThrow();
        }
        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0];
        return new Shape[]{new Shape(in.get(0), in.get(1), in.get(2) * 2, in.get(3) * 2)};
    }
}
