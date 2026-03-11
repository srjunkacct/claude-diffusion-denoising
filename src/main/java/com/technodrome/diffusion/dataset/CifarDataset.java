package com.technodrome.diffusion.dataset;

import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;

/**
 * CIFAR-10 dataset wrapper.
 * Wraps DJL's built-in Cifar10 dataset, normalizes to [-1, 1] in NCHW format.
 *
 * CIFAR-10: 50,000 training images, 32x32x3.
 */
public class CifarDataset {

    /**
     * Create the CIFAR-10 training dataset.
     *
     * @param batchSize  batch size
     * @param shuffle    whether to shuffle
     * @return DJL Dataset ready for iteration
     */
    public static Dataset getTrainDataset(int batchSize, boolean shuffle) {
        // DJL Cifar10 returns images as [H, W, C] float32 in [0, 255]
        // Convert to NCHW [-1, 1] for PyTorch
        Pipeline pipeline = new Pipeline();
        pipeline.add(new ToNCHW());
        pipeline.add(new NormalizeToMinusOneOne());

        Cifar10.Builder builder = Cifar10.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, shuffle)
                .optPipeline(pipeline);

        return builder.build();
    }

    public static Dataset getTestDataset(int batchSize) {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new ToNCHW());
        pipeline.add(new NormalizeToMinusOneOne());

        return Cifar10.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optPipeline(pipeline)
                .build();
    }

    /**
     * Transform [H, W, C] -> [C, H, W] for PyTorch NCHW convention.
     */
    private static class ToNCHW implements Transform {
        @Override
        public NDArray transform(NDArray array) {
            // Handles both [H, W, C] and batched [B, H, W, C]
            if (array.getShape().dimension() == 4) {
                return array.transpose(0, 3, 1, 2);
            }
            return array.transpose(2, 0, 1);
        }
    }

    private static class NormalizeToMinusOneOne implements Transform {
        @Override
        public NDArray transform(NDArray array) {
            // DJL Cifar10 provides float32 [0, 255] (NOT [0, 1])
            // [0, 255] -> [-1, 1]: x / 127.5 - 1
            return array.div(127.5f).sub(1.0f);
        }
    }
}
