package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;

/**
 * Dense (fully connected) layer with variance-scaling initialization.
 * Ported from nn.py: dense().
 */
public class DenseBlock extends AbstractBlock {

    private final Linear linear;

    public DenseBlock(int outFeatures, float initScale, boolean useBias) {
        linear = addChildBlock("linear", Linear.builder()
                .setUnits(outFeatures)
                .optBias(useBias)
                .build());
        linear.setInitializer(new VarianceScalingInitializer(initScale), Parameter.Type.WEIGHT);
        if (useBias) {
            linear.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        }
    }

    /** Convenience: bias=true, initScale=1 */
    public DenseBlock(int outFeatures) {
        this(outFeatures, 1.0f, true);
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        linear.initialize(manager, dataType, inputShapes);
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        return linear.forward(ps, inputs, training);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return linear.getOutputShapes(inputShapes);
    }
}
