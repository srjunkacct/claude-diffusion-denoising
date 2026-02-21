package com.technodrome.diffusion.training;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.technodrome.diffusion.util.ImageUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;
import java.util.function.BiFunction;

/**
 * Manual training loop for diffusion models.
 * Handles: GradientCollector, gradient clipping, optimizer step, EMA, checkpointing,
 * sample generation.
 *
 * Ported from tpu_utils.py: run_training() and the training infrastructure.
 */
public class DiffusionTrainer {

    private static final Logger logger = LoggerFactory.getLogger(DiffusionTrainer.class);

    public static class Config {
        public float lr = 2e-4f;
        public int warmup = 5000;
        public float gradClip = 1.0f;
        public float emaDecay = 0.9999f;
        public int logInterval = 100;
        public int sampleInterval = 5000;
        public int saveInterval = 10000;
        public int keepCheckpointMax = 2;
        public int sampleBatchSize = 16;
        public Path outputDir = Path.of("output");
        public boolean randflip = true;
    }

    private final Block model;
    private final Optimizer optimizer;
    private final EMAHelper ema;
    private final CheckpointManager checkpointManager;
    private final Config config;
    private final Random random = new Random();
    private int globalStep = 0;

    public DiffusionTrainer(Block model, Config config) {
        this.model = model;
        this.config = config;

        // Adam optimizer with warmup LR schedule
        Tracker lrTracker = LRScheduler.createWarmupTracker(config.lr, config.warmup);
        this.optimizer = Adam.builder()
                .optLearningRateTracker(lrTracker)
                .optEpsilon(1e-8f)
                .build();

        this.ema = new EMAHelper(model, config.emaDecay);
        this.checkpointManager = new CheckpointManager(
                config.outputDir.resolve("checkpoints"), config.keepCheckpointMax);
    }

    /**
     * Run the training loop.
     *
     * @param dataset      training dataset
     * @param lossFn       function (xStart [B,C,H,W], t [B]) -> per-sample losses [B]
     * @param sampleFn     function (manager, sampleShape) -> generated samples
     * @param totalSteps   total training steps
     * @param imageShape   shape of a single sample for generation [C, H, W]
     */
    public void train(Dataset dataset, BiFunction<NDArray, NDArray, NDArray> lossFn,
                      BiFunction<NDManager, Shape, NDArray> sampleFn,
                      int totalSteps, Shape imageShape) throws Exception {
        NDManager manager = NDManager.newBaseManager();
        Device device = manager.getDevice();

        // Initialize model
        Shape inputShape = new Shape(config.sampleBatchSize, imageShape.get(0),
                imageShape.get(1), imageShape.get(2));
        model.initialize(manager, DataType.FLOAT32, inputShape, new Shape(config.sampleBatchSize));

        ParameterStore ps = new ParameterStore(manager, false);
        ema.register(manager);

        // Try to load latest checkpoint
        try {
            int loadedStep = checkpointManager.loadLatest(model, manager);
            if (loadedStep >= 0) {
                globalStep = loadedStep;
                logger.info("Resumed from step {}", globalStep);
            }
        } catch (Exception e) {
            logger.info("No checkpoint to resume from, starting fresh");
        }

        logger.info("Starting training from step {} to {}", globalStep, totalSteps);
        Path samplesDir = config.outputDir.resolve("samples");
        java.nio.file.Files.createDirectories(samplesDir);

        while (globalStep < totalSteps) {
            for (Batch batch : dataset.getData(manager)) {
                if (globalStep >= totalSteps) break;

                try (NDManager stepManager = manager.newSubManager()) {
                    NDArray xStart = batch.getData().singletonOrThrow().toDevice(device, false);

                    // Random horizontal flip
                    if (config.randflip) {
                        xStart = randomFlipLR(xStart, stepManager);
                    }

                    long batchSize = xStart.getShape().get(0);

                    // Random timesteps
                    int[] tArr = new int[(int) batchSize];
                    int numTimesteps = getNumTimesteps(lossFn);
                    for (int i = 0; i < tArr.length; i++) {
                        tArr[i] = random.nextInt(1000); // default T=1000
                    }
                    NDArray t = stepManager.create(tArr);

                    // Forward pass + loss
                    NDArray loss;
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray losses = lossFn.apply(xStart, t);
                        loss = losses.mean();
                        gc.backward(loss);
                    }

                    // Gradient clipping
                    clipGradients(config.gradClip);

                    // Optimizer step - update each parameter individually
                    for (var pair : model.getParameters()) {
                        if (pair.getValue().requiresGradient() && pair.getValue().getArray().hasGradient()) {
                            NDArray weight = pair.getValue().getArray();
                            NDArray grad = weight.getGradient();
                            optimizer.update(pair.getKey(), weight, grad);
                        }
                    }

                    // EMA update
                    ema.update();

                    globalStep++;

                    // Logging
                    if (globalStep % config.logInterval == 0) {
                        logger.info("Step {}: loss = {:.6f}", globalStep, loss.getFloat());
                    }

                    // Sample generation
                    if (globalStep % config.sampleInterval == 0) {
                        generateAndSaveSamples(manager, sampleFn, imageShape, samplesDir);
                    }

                    // Checkpoint
                    if (globalStep % config.saveInterval == 0) {
                        checkpointManager.save(model, globalStep);
                    }
                }
                batch.close();
            }
        }

        // Final save
        checkpointManager.save(model, globalStep);
        manager.close();
        logger.info("Training complete at step {}", globalStep);
    }

    private void generateAndSaveSamples(NDManager manager,
                                         BiFunction<NDManager, Shape, NDArray> sampleFn,
                                         Shape imageShape, Path samplesDir) {
        logger.info("Generating samples at step {}...", globalStep);
        try (NDManager sampleManager = manager.newSubManager()) {
            Shape sampleShape = new Shape(config.sampleBatchSize,
                    imageShape.get(0), imageShape.get(1), imageShape.get(2));

            // Use EMA parameters for sampling
            var origParams = ema.swapToEma();
            NDArray samples = sampleFn.apply(sampleManager, sampleShape);
            ema.restoreFromEma(origParams);

            // Save tiled image
            String filename = String.format("samples_%d.png", globalStep);
            ImageUtils.saveTiledImages(samplesDir.resolve(filename), samples);
            logger.info("Samples saved to {}", filename);
        } catch (Exception e) {
            logger.error("Failed to generate samples", e);
        }
    }

    private NDArray randomFlipLR(NDArray x, NDManager manager) {
        // Random horizontal flip with 50% probability per image
        Shape shape = x.getShape();
        long B = shape.get(0);
        NDArray flipped = x.flip(3); // flip along W dimension (NCHW)
        NDArray mask = manager.randomUniform(0, 1, new Shape(B, 1, 1, 1))
                .lt(0.5f).toType(DataType.FLOAT32, false);
        return mask.mul(flipped).add(mask.neg().add(1).mul(x));
    }

    private void clipGradients(float maxNorm) {
        // Compute global gradient norm
        float totalNormSq = 0;
        for (var pair : model.getParameters()) {
            if (pair.getValue().requiresGradient() && pair.getValue().getArray().hasGradient()) {
                NDArray grad = pair.getValue().getArray().getGradient();
                totalNormSq += grad.square().sum().getFloat();
            }
        }
        float totalNorm = (float) Math.sqrt(totalNormSq);

        // Scale gradients if norm exceeds threshold
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (var pair : model.getParameters()) {
                if (pair.getValue().requiresGradient() && pair.getValue().getArray().hasGradient()) {
                    pair.getValue().getArray().getGradient().muli(scale);
                }
            }
        }
    }

    private int getNumTimesteps(Object lossFn) {
        return 1000; // Default, overridden by diffusion class
    }

    public int getGlobalStep() {
        return globalStep;
    }

    // Inner static class for Engine access
    private static class Engine {
        static ai.djl.engine.Engine getInstance() {
            return ai.djl.engine.Engine.getInstance();
        }
    }
}
