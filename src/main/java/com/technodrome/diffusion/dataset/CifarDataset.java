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
        // DJL Cifar10 returns images as [C, H, W] float32 in [0, 1]
        // We need to transform to [-1, 1]
        Pipeline pipeline = new Pipeline();
        pipeline.add(new NormalizeToMinusOneOne());

        Cifar10.Builder builder = Cifar10.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, shuffle)
                .optPipeline(pipeline);

        return builder.build();
    }

    public static Dataset getTestDataset(int batchSize) {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new NormalizeToMinusOneOne());

        return Cifar10.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optPipeline(pipeline)
                .build();
    }

    /**
     * Transform that converts [0, 1] float to [-1, 1] float.
     * DJL's Cifar10 already provides [0, 1] normalized images.
     */
    private static class NormalizeToMinusOneOne implements Transform {
        @Override
        public NDArray transform(NDArray array) {
            // [0, 1] -> [-1, 1]: x * 2 - 1
            return array.mul(2.0f).sub(1.0f);
        }
    }
}
