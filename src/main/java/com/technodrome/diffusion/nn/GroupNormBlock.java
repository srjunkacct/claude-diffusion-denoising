package com.technodrome.diffusion.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;

/**
 * Group Normalization (Wu & He, 2018).
 * DJL lacks built-in GroupNorm, so we implement it from scratch.
 * Input: [B, C, H, W] in NCHW format.
 * Ported from unet.py: normalize() -> tf.contrib.layers.group_norm (default 32 groups).
 */
public class GroupNormBlock extends AbstractBlock {

    private static final float EPS = 1e-5f;

    private final int numGroups;
    private final int numChannels;
    private final Parameter gamma;
    private final Parameter bias;

    public GroupNormBlock(int numGroups, int numChannels) {
        this.numGroups = numGroups;
        this.numChannels = numChannels;

        gamma = addParameter(
                Parameter.builder()
                        .setName("gamma")
                        .setType(Parameter.Type.GAMMA)
                        .optShape(new Shape(numChannels))
                        .build());
        bias = addParameter(
                Parameter.builder()
                        .setName("beta")
                        .setType(Parameter.Type.BETA)
                        .optShape(new Shape(numChannels))
                        .build());

        setInitializer(Initializer.ONES, Parameter.Type.GAMMA);
        setInitializer(Initializer.ZEROS, Parameter.Type.BETA);
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        NDArray g = ps.getValue(gamma, x.getDevice(), training);
        NDArray b = ps.getValue(bias, x.getDevice(), training);

        Shape shape = x.getShape();
        long batchSize = shape.get(0);
        long C = shape.get(1);
        long H = shape.get(2);
        long W = shape.get(3);
        int G = numGroups;

        // Reshape to [B, G, C/G, H, W]
        NDArray y = x.reshape(batchSize, G, C / G, H, W);

        // Compute mean and variance over (C/G, H, W) dimensions
        // DJL PyTorch only supports single-axis mean, so flatten then reduce
        NDArray flat = y.reshape(batchSize, G, -1); // [B, G, (C/G)*H*W]
        NDArray mean = flat.mean(new int[]{2}, true); // [B, G, 1]
        mean = mean.reshape(batchSize, G, 1, 1, 1);  // broadcast back to [B, G, C/G, H, W]
        NDArray diff = y.sub(mean);
        NDArray diffFlat = diff.reshape(batchSize, G, -1);
        NDArray var = diffFlat.square().mean(new int[]{2}, true); // [B, G, 1]
        var = var.reshape(batchSize, G, 1, 1, 1);

        // Normalize
        y = diff.div(var.add(EPS).sqrt());

        // Reshape back to [B, C, H, W]
        y = y.reshape(batchSize, C, H, W);

        // Scale and shift: gamma [C] -> [1, C, 1, 1], beta [C] -> [1, C, 1, 1]
        // Close reshaped views explicitly — they are created on the parameter's (parent) manager
        // and would otherwise leak, accumulating ~102 views per training step.
        try (NDArray gReshaped = g.reshape(1, C, 1, 1);
             NDArray bReshaped = b.reshape(1, C, 1, 1)) {
            y = y.mul(gReshaped).add(bReshaped);
        }

        return new NDList(y);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }
}
