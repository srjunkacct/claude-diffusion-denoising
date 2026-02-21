package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Network-in-Network: 1x1 convolution.
 * Ported from nn.py: nin().
 * In NHWC the original uses contract_inner (einsum on last axis);
 * in NCHW a 1x1 Conv2d is mathematically equivalent.
 */
public class NinBlock extends AbstractBlock {

    private final Conv2dBlock conv;

    public NinBlock(int outChannels, float initScale) {
        conv = addChildBlock("nin", new Conv2dBlock(outChannels, 1, 1, initScale, true));
    }

    public NinBlock(int outChannels) {
        this(outChannels, 1.0f);
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        return conv.forward(ps, inputs, training);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return conv.getOutputShapes(inputShapes);
    }
}
