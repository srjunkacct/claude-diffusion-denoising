package com.technodrome.diffusion.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.technodrome.diffusion.nn.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Full U-Net with encoder, middle bottleneck, and decoder with skip connections.
 * Ported from unet.py: model().
 *
 * Input: NDList[x: [B, inCh, H, W], t: [B] (int timesteps)]
 * Output: NDList[result: [B, outCh, H, W]]
 *
 * Key NHWC->NCHW changes:
 *   - Skip connection concat on axis=1 (channel dim) instead of axis=-1
 *   - Spatial resolution check uses shape.get(2) instead of shape[1]
 *   - Timestep embedding broadcast: [B, C, 1, 1] instead of [B, 1, 1, C]
 */
public class UNetBlock extends AbstractBlock {

    private final int ch;
    private final int[] chMult;
    private final int numResolutions;
    private final int numResBlocks;
    private final int imageSize;

    // Timestep embedding
    private final DenseBlock tembDense0;
    private final DenseBlock tembDense1;

    // Initial conv
    private final Conv2dBlock convIn;

    // Encoder: downResBlocks[level][block]
    private final ResnetBlock[][] downResBlocks;
    private final AttnBlock[][] downAttnBlocks; // null array entry if no attn at level
    private final DownsampleBlock[] downsamples;  // null entry at last level

    // Middle
    private final ResnetBlock midBlock1;
    private final AttnBlock midAttn;
    private final ResnetBlock midBlock2;

    // Decoder: upResBlocks[level][block]
    private final ResnetBlock[][] upResBlocks;
    private final AttnBlock[][] upAttnBlocks;
    private final UpsampleBlock[] upsamples; // null entry at level 0

    // End
    private final GroupNormBlock normOut;
    private final Conv2dBlock convOut;

    /**
     * @param ch               base channel count (e.g. 128)
     * @param outCh            output channels (e.g. 3 for RGB, 6 for learned variance)
     * @param chMult           channel multipliers per level (e.g. {1,2,2,2})
     * @param numResBlocks     residual blocks per level (e.g. 2)
     * @param attnResolutions  spatial resolutions where attention is applied (e.g. {16})
     * @param dropout          dropout rate
     * @param resampWithConv   use conv for up/downsample (vs pool/resize)
     * @param imageSize        input spatial size (e.g. 32 for CIFAR, 256 for CelebA-HQ)
     */
    public UNetBlock(int ch, int outCh, int[] chMult, int numResBlocks,
                     int[] attnResolutions, float dropout, boolean resampWithConv,
                     int imageSize) {
        this.ch = ch;
        this.chMult = chMult.clone();
        this.numResolutions = chMult.length;
        this.numResBlocks = numResBlocks;
        this.imageSize = imageSize;

        int tembDim = ch * 4;

        // --- Timestep embedding ---
        tembDense0 = addChildBlock("temb_dense0", new DenseBlock(tembDim));
        tembDense1 = addChildBlock("temb_dense1", new DenseBlock(tembDim));

        // --- Encoder ---
        convIn = addChildBlock("conv_in", new Conv2dBlock(ch));

        // Precompute which levels use attention
        boolean[] useAttn = new boolean[numResolutions];
        for (int i = 0; i < numResolutions; i++) {
            int res = imageSize >> i;
            for (int ar : attnResolutions) {
                if (res == ar) { useAttn[i] = true; break; }
            }
        }

        // Track skip connection channels for decoder concat computation
        List<Integer> skipChannels = new ArrayList<>();
        skipChannels.add(ch); // conv_in output

        downResBlocks = new ResnetBlock[numResolutions][numResBlocks];
        downAttnBlocks = new AttnBlock[numResolutions][];
        downsamples = new DownsampleBlock[numResolutions];

        int currentCh = ch;
        for (int iLevel = 0; iLevel < numResolutions; iLevel++) {
            int levelCh = ch * chMult[iLevel];

            if (useAttn[iLevel]) {
                downAttnBlocks[iLevel] = new AttnBlock[numResBlocks];
            }

            for (int iBlock = 0; iBlock < numResBlocks; iBlock++) {
                downResBlocks[iLevel][iBlock] = addChildBlock(
                        String.format("down_%d_block_%d", iLevel, iBlock),
                        new ResnetBlock(currentCh, levelCh, tembDim, dropout, false));
                currentCh = levelCh;

                if (useAttn[iLevel]) {
                    downAttnBlocks[iLevel][iBlock] = addChildBlock(
                            String.format("down_%d_attn_%d", iLevel, iBlock),
                            new AttnBlock(currentCh));
                }
                skipChannels.add(currentCh);
            }

            if (iLevel != numResolutions - 1) {
                downsamples[iLevel] = addChildBlock(
                        String.format("down_%d_downsample", iLevel),
                        new DownsampleBlock(currentCh, resampWithConv));
                skipChannels.add(currentCh);
            }
        }

        // --- Middle ---
        midBlock1 = addChildBlock("mid_block_1",
                new ResnetBlock(currentCh, currentCh, tembDim, dropout, false));
        midAttn = addChildBlock("mid_attn_1", new AttnBlock(currentCh));
        midBlock2 = addChildBlock("mid_block_2",
                new ResnetBlock(currentCh, currentCh, tembDim, dropout, false));

        // --- Decoder ---
        upResBlocks = new ResnetBlock[numResolutions][numResBlocks + 1];
        upAttnBlocks = new AttnBlock[numResolutions][];
        upsamples = new UpsampleBlock[numResolutions];

        for (int iLevel = numResolutions - 1; iLevel >= 0; iLevel--) {
            int levelCh = ch * chMult[iLevel];

            if (useAttn[iLevel]) {
                upAttnBlocks[iLevel] = new AttnBlock[numResBlocks + 1];
            }

            for (int iBlock = 0; iBlock < numResBlocks + 1; iBlock++) {
                int skipCh = skipChannels.remove(skipChannels.size() - 1);
                int concatCh = currentCh + skipCh;

                upResBlocks[iLevel][iBlock] = addChildBlock(
                        String.format("up_%d_block_%d", iLevel, iBlock),
                        new ResnetBlock(concatCh, levelCh, tembDim, dropout, false));
                currentCh = levelCh;

                if (useAttn[iLevel]) {
                    upAttnBlocks[iLevel][iBlock] = addChildBlock(
                            String.format("up_%d_attn_%d", iLevel, iBlock),
                            new AttnBlock(currentCh));
                }
            }

            if (iLevel != 0) {
                upsamples[iLevel] = addChildBlock(
                        String.format("up_%d_upsample", iLevel),
                        new UpsampleBlock(currentCh, resampWithConv));
            }
        }

        assert skipChannels.isEmpty() : "Skip connection channel mismatch";

        // --- End ---
        normOut = addChildBlock("norm_out", new GroupNormBlock(32, currentCh));
        convOut = addChildBlock("conv_out", new Conv2dBlock(outCh, 3, 1, 0.0f, true));
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.get(0); // [B, C, H, W]
        NDArray t = inputs.get(1); // [B] int timesteps

        // --- Timestep embedding ---
        NDArray temb = TimestepEmbedding.getTimestepEmbedding(t, ch);
        temb = tembDense0.forward(ps, new NDList(temb), training).singletonOrThrow();
        temb = temb.mul(temb.getNDArrayInternal().sigmoid()); // swish
        temb = tembDense1.forward(ps, new NDList(temb), training).singletonOrThrow();
        // temb: [B, ch*4]

        // --- Encoder ---
        List<NDArray> hs = new ArrayList<>();
        NDArray h = convIn.forward(ps, new NDList(x), training).singletonOrThrow();
        hs.add(h);

        for (int iLevel = 0; iLevel < numResolutions; iLevel++) {
            for (int iBlock = 0; iBlock < numResBlocks; iBlock++) {
                h = downResBlocks[iLevel][iBlock]
                        .forward(ps, new NDList(h, temb), training).singletonOrThrow();
                if (downAttnBlocks[iLevel] != null) {
                    h = downAttnBlocks[iLevel][iBlock]
                            .forward(ps, new NDList(h), training).singletonOrThrow();
                }
                hs.add(h);
            }
            if (downsamples[iLevel] != null) {
                h = downsamples[iLevel]
                        .forward(ps, new NDList(h), training).singletonOrThrow();
                hs.add(h);
            }
        }

        // --- Middle ---
        h = midBlock1.forward(ps, new NDList(h, temb), training).singletonOrThrow();
        h = midAttn.forward(ps, new NDList(h), training).singletonOrThrow();
        h = midBlock2.forward(ps, new NDList(h, temb), training).singletonOrThrow();

        // --- Decoder ---
        for (int iLevel = numResolutions - 1; iLevel >= 0; iLevel--) {
            for (int iBlock = 0; iBlock < numResBlocks + 1; iBlock++) {
                NDArray skip = hs.remove(hs.size() - 1);
                h = NDArrays.concat(new NDList(h, skip), 1); // NCHW: concat on channel dim
                h = upResBlocks[iLevel][iBlock]
                        .forward(ps, new NDList(h, temb), training).singletonOrThrow();
                if (upAttnBlocks[iLevel] != null) {
                    h = upAttnBlocks[iLevel][iBlock]
                            .forward(ps, new NDList(h), training).singletonOrThrow();
                }
            }
            if (upsamples[iLevel] != null) {
                h = upsamples[iLevel]
                        .forward(ps, new NDList(h), training).singletonOrThrow();
            }
        }

        assert hs.isEmpty() : "Not all skip connections consumed";

        // --- End ---
        h = normOut.forward(ps, new NDList(h), training).singletonOrThrow();
        h = h.mul(h.getNDArrayInternal().sigmoid()); // swish
        h = convOut.forward(ps, new NDList(h), training).singletonOrThrow();

        return new NDList(h);
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType,
                                          Shape... inputShapes) {
        Shape xShape = inputShapes[0]; // [B, inCh, H, W]
        long B = xShape.get(0);
        int tembDim = ch * 4;

        // --- Timestep embedding ---
        Shape tembInShape = new Shape(B, ch);
        tembDense0.initialize(manager, dataType, tembInShape);
        Shape tembShape = new Shape(B, tembDim);
        tembDense1.initialize(manager, dataType, tembShape);

        // --- Encoder ---
        convIn.initialize(manager, dataType, xShape);
        long curH = xShape.get(2), curW = xShape.get(3);

        // Track skip connection channel counts for decoder
        List<Integer> skipChannels = new ArrayList<>();
        skipChannels.add(ch);

        int currentCh = ch;
        for (int iLevel = 0; iLevel < numResolutions; iLevel++) {
            int levelCh = ch * chMult[iLevel];

            for (int iBlock = 0; iBlock < numResBlocks; iBlock++) {
                Shape hShape = new Shape(B, currentCh, curH, curW);
                downResBlocks[iLevel][iBlock].initialize(manager, dataType, hShape, tembShape);
                currentCh = levelCh;

                if (downAttnBlocks[iLevel] != null) {
                    Shape attnShape = new Shape(B, currentCh, curH, curW);
                    downAttnBlocks[iLevel][iBlock].initialize(manager, dataType, attnShape);
                }
                skipChannels.add(currentCh);
            }

            if (downsamples[iLevel] != null) {
                Shape dsShape = new Shape(B, currentCh, curH, curW);
                downsamples[iLevel].initialize(manager, dataType, dsShape);
                curH /= 2;
                curW /= 2;
                skipChannels.add(currentCh);
            }
        }

        // --- Middle ---
        Shape midShape = new Shape(B, currentCh, curH, curW);
        midBlock1.initialize(manager, dataType, midShape, tembShape);
        midAttn.initialize(manager, dataType, midShape);
        midBlock2.initialize(manager, dataType, midShape, tembShape);

        // --- Decoder ---
        for (int iLevel = numResolutions - 1; iLevel >= 0; iLevel--) {
            int levelCh = ch * chMult[iLevel];

            for (int iBlock = 0; iBlock < numResBlocks + 1; iBlock++) {
                int skipCh = skipChannels.remove(skipChannels.size() - 1);
                int concatCh = currentCh + skipCh;

                Shape concatShape = new Shape(B, concatCh, curH, curW);
                upResBlocks[iLevel][iBlock].initialize(manager, dataType, concatShape, tembShape);
                currentCh = levelCh;

                if (upAttnBlocks[iLevel] != null) {
                    Shape attnShape = new Shape(B, currentCh, curH, curW);
                    upAttnBlocks[iLevel][iBlock].initialize(manager, dataType, attnShape);
                }
            }

            if (upsamples[iLevel] != null) {
                Shape usShape = new Shape(B, currentCh, curH, curW);
                upsamples[iLevel].initialize(manager, dataType, usShape);
                curH *= 2;
                curW *= 2;
            }
        }

        // --- End ---
        Shape endShape = new Shape(B, currentCh, curH, curW);
        normOut.initialize(manager, dataType, endShape);
        convOut.initialize(manager, dataType, endShape);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0]; // [B, inCh, H, W]
        long outCh = convOut.getOutputShapes(
                new Shape[]{new Shape(in.get(0), ch, in.get(2), in.get(3))}
        )[0].get(1);
        return new Shape[]{new Shape(in.get(0), outCh, in.get(2), in.get(3))};
    }
}
