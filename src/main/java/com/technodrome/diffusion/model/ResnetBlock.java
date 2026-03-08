package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.technodrome.diffusion.nn.*;

/**
 * Residual block with GroupNorm, Swish, Conv, timestep embedding injection, and skip connection.
 * Ported from unet.py: resnet_block().
 *
 * Input: NDList[x: [B, inCh, H, W], temb: [B, tembDim]]
 * Output: NDList[result: [B, outCh, H, W]]
 */
public class ResnetBlock extends AbstractBlock {

    private final int outCh;
    private final boolean useShortcut;
    private final boolean convShortcut;

    private final GroupNormBlock norm1;
    private final SwishBlock swish1;
    private final Conv2dBlock conv1;
    private final DenseBlock tembProj;
    private final GroupNormBlock norm2;
    private final SwishBlock swish2;
    private final Dropout dropout;
    private final Conv2dBlock conv2;
    private AbstractBlock shortcut; // NinBlock or Conv2dBlock, if needed

    /**
     * @param inCh         input channels
     * @param outCh        output channels
     * @param tembDim      timestep embedding dimension (ch * 4)
     * @param dropoutRate  dropout rate
     * @param convShortcut if true, use 3x3 conv for shortcut; else use 1x1 NIN
     */
    public ResnetBlock(int inCh, int outCh, int tembDim, float dropoutRate, boolean convShortcut) {
        this.outCh = outCh;
        this.useShortcut = (inCh != outCh);
        this.convShortcut = convShortcut;

        norm1 = addChildBlock("norm1", new GroupNormBlock(32, inCh));
        swish1 = addChildBlock("swish1", new SwishBlock());
        conv1 = addChildBlock("conv1", new Conv2dBlock(outCh));

        // Timestep embedding projection: dense(swish(temb)) -> [B, outCh]
        tembProj = addChildBlock("temb_proj", new DenseBlock(outCh));

        norm2 = addChildBlock("norm2", new GroupNormBlock(32, outCh));
        swish2 = addChildBlock("swish2", new SwishBlock());
        dropout = addChildBlock("dropout", Dropout.builder().optRate(dropoutRate).build());
        conv2 = addChildBlock("conv2", new Conv2dBlock(outCh, 3, 1, 0.0f, true)); // init_scale=0

        if (useShortcut) {
            if (convShortcut) {
                shortcut = addChildBlock("shortcut", new Conv2dBlock(outCh));
            } else {
                shortcut = addChildBlock("shortcut", new NinBlock(outCh));
            }
        }
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        // inputShapes: [xShape: [B, inCh, H, W], tembShape: [B, tembDim]]
        Shape xShape = inputShapes[0];
        Shape tembShape = inputShapes[1];

        // norm1, swish1, conv1 all take xShape
        norm1.initialize(manager, dataType, xShape);
        swish1.initialize(manager, dataType, xShape);
        conv1.initialize(manager, dataType, xShape);

        // tembProj takes [B, tembDim]
        tembProj.initialize(manager, dataType, tembShape);

        // After conv1: [B, outCh, H, W]
        Shape hShape = new Shape(xShape.get(0), outCh, xShape.get(2), xShape.get(3));
        norm2.initialize(manager, dataType, hShape);
        swish2.initialize(manager, dataType, hShape);
        dropout.initialize(manager, dataType, hShape);
        conv2.initialize(manager, dataType, hShape);

        if (useShortcut) {
            shortcut.initialize(manager, dataType, xShape);
        }
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.get(0);    // [B, inCh, H, W]
        NDArray temb = inputs.get(1); // [B, tembDim]

        NDArray h = x;

        // GroupNorm -> Swish -> Conv
        h = norm1.forward(ps, new NDList(h), training).singletonOrThrow();
        h = swish1.forward(ps, new NDList(h), training).singletonOrThrow();
        h = conv1.forward(ps, new NDList(h), training).singletonOrThrow();

        // Add timestep embedding: dense(swish(temb)) broadcast to spatial dims
        // temb: [B, tembDim] -> swish -> dense -> [B, outCh] -> [B, outCh, 1, 1]
        NDArray tembSwish = temb.mul(temb.getNDArrayInternal().sigmoid()); // swish
        NDArray tembOut = tembProj.forward(ps, new NDList(tembSwish), training).singletonOrThrow();
        h = h.add(tembOut.reshape(tembOut.getShape().get(0), outCh, 1, 1));

        // GroupNorm -> Swish -> Dropout -> Conv (init_scale=0)
        h = norm2.forward(ps, new NDList(h), training).singletonOrThrow();
        h = swish2.forward(ps, new NDList(h), training).singletonOrThrow();
        h = dropout.forward(ps, new NDList(h), training).singletonOrThrow();
        h = conv2.forward(ps, new NDList(h), training).singletonOrThrow();

        // Skip connection
        if (useShortcut) {
            x = shortcut.forward(ps, new NDList(x), training).singletonOrThrow();
        }

        return new NDList(x.add(h));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0];
        return new Shape[]{new Shape(in.get(0), outCh, in.get(2), in.get(3))};
    }
}
