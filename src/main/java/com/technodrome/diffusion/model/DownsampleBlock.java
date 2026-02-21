package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.technodrome.diffusion.nn.Conv2dBlock;

/**
 * 2x spatial downsampling via stride-2 conv or average pooling.
 * Ported from unet.py: downsample().
 * Input/output: NCHW [B, C, H, W] -> [B, C, H/2, W/2].
 */
public class DownsampleBlock extends AbstractBlock {

    private final boolean withConv;
    private Conv2dBlock conv;

    public DownsampleBlock(int channels, boolean withConv) {
        this.withConv = withConv;
        if (withConv) {
            conv = addChildBlock("conv", new Conv2dBlock(channels, 3, 2, 1.0f, true));
        }
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        if (withConv) {
            return conv.forward(ps, new NDList(x), training);
        } else {
            // Average pooling 2x2 with stride 2
            return new NDList(Pool.avgPool2d(x, new Shape(2, 2), new Shape(2, 2), new Shape(0, 0), false, false));
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0];
        return new Shape[]{new Shape(in.get(0), in.get(1), in.get(2) / 2, in.get(3) / 2)};
    }
}
