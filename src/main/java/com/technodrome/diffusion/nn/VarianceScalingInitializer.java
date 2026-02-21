package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.initializer.Initializer;

/**
 * Variance scaling initializer matching TF's variance_scaling(mode='fan_avg', distribution='uniform').
 * Ported from nn.py: default_init(scale).
 */
public class VarianceScalingInitializer implements Initializer {

    private final double scale;

    public VarianceScalingInitializer(double scale) {
        // When scale==0, TF uses 1e-10 (effectively zero init)
        this.scale = (scale == 0.0) ? 1e-10 : scale;
    }

    @Override
    public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
        long[] dims = shape.getShape();
        double fanIn;
        double fanOut;

        if (dims.length == 1) {
            // Bias vector
            fanIn = dims[0];
            fanOut = dims[0];
        } else if (dims.length == 2) {
            // Linear weight [outFeatures, inFeatures] (PyTorch convention)
            fanIn = dims[1];
            fanOut = dims[0];
        } else if (dims.length >= 3) {
            // Conv weight [outChannels, inChannels, kH, kW, ...]
            long receptiveField = 1;
            for (int i = 2; i < dims.length; i++) {
                receptiveField *= dims[i];
            }
            fanIn = dims[1] * receptiveField;
            fanOut = dims[0] * receptiveField;
        } else {
            fanIn = 1;
            fanOut = 1;
        }

        // fan_avg mode, uniform distribution
        double fanAvg = (fanIn + fanOut) / 2.0;
        double limit = Math.sqrt(3.0 * scale / fanAvg);
        return manager.randomUniform((float) -limit, (float) limit, shape, dataType);
    }
}
