package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.nn.Parameter;
import ai.djl.util.PairList;

/**
 * Conv2d with variance-scaling initialization.
 * Ported from nn.py: conv2d().
 * NCHW layout (PyTorch default).
 */
public class Conv2dBlock extends AbstractBlock {

    private final Conv2d conv;

    public Conv2dBlock(int outChannels, int kernelSize, int stride, float initScale, boolean useBias) {
        int pad = (kernelSize - 1) / 2; // 'SAME' padding for odd kernel sizes
        conv = addChildBlock("conv", Conv2d.builder()
                .setFilters(outChannels)
                .setKernelShape(new Shape(kernelSize, kernelSize))
                .optStride(new Shape(stride, stride))
                .optPadding(new Shape(pad, pad))
                .optBias(useBias)
                .build());
        conv.setInitializer(new VarianceScalingInitializer(initScale), Parameter.Type.WEIGHT);
        if (useBias) {
            conv.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        }
    }

    /** Convenience: 3x3, stride 1, bias=true, initScale=1 */
    public Conv2dBlock(int outChannels) {
        this(outChannels, 3, 1, 1.0f, true);
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
