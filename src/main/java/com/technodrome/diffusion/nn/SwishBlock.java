package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Swish activation: x * sigmoid(x).
 * Ported from unet.py: nonlinearity().
 */
public class SwishBlock extends AbstractBlock {

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        return new NDList(x.mul(x.getNDArrayInternal().sigmoid()));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }
}
