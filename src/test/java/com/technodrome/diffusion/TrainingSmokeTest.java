package com.technodrome.diffusion;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.technodrome.diffusion.diffusion.BetaSchedule;
import com.technodrome.diffusion.diffusion.GaussianDiffusion2;
import com.technodrome.diffusion.model.UNetBlock;
import org.junit.jupiter.api.Test;

import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke test: verifies the training pipeline runs end-to-end
 * (forward pass, loss computation, backward pass, parameter update)
 * with a tiny model and synthetic data.
 */
class TrainingSmokeTest {

    @Test
    void trainingSmokeTest() {
        // --- Tiny config (ch must be >= 32 for GroupNorm with 32 groups) ---
        int ch = 32;
        int[] chMult = {1, 2};
        int numResBlocks = 1;
        int[] attnResolutions = {8};
        int imageSize = 16;
        float dropout = 0f;
        int outCh = 3;
        int numTimesteps = 10;
        int batchSize = 2;
        int numSteps = 5;

        // --- Beta schedule & diffusion ---
        double[] betas = BetaSchedule.getBetaSchedule("linear", 0.0001, 0.02, numTimesteps);
        GaussianDiffusion2 diffusion = new GaussianDiffusion2(betas, "eps", "fixedlarge", "mse");

        // --- Model ---
        UNetBlock model = new UNetBlock(ch, outCh, chMult, numResBlocks,
                attnResolutions, dropout, true, imageSize);

        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            Shape xShape = new Shape(batchSize, 3, imageSize, imageSize);
            Shape tShape = new Shape(batchSize);
            model.initialize(manager, DataType.FLOAT32, xShape, tShape);

            ParameterStore ps = new ParameterStore(manager, false);

            // --- Verify forward pass output shape ---
            NDArray dummyX = manager.randomNormal(xShape);
            NDArray dummyT = manager.zeros(tShape).toType(DataType.INT32, false);
            NDArray output = model.forward(ps, new NDList(dummyX, dummyT), false).singletonOrThrow();
            assertEquals(new Shape(batchSize, 3, imageSize, imageSize), output.getShape(),
                    "Model output shape should be [B, 3, 16, 16]");

            // --- Optimizer ---
            Optimizer optimizer = Adam.builder()
                    .optLearningRateTracker(Tracker.fixed(2e-4f))
                    .optEpsilon(1e-8f)
                    .build();

            // --- Denoise function ---
            BiFunction<NDArray, NDArray, NDArray> denoiseFn = (xT, t) ->
                    model.forward(ps, new NDList(xT, t), true).singletonOrThrow();

            // --- Training loop ---
            float[] losses = new float[numSteps];

            for (int step = 0; step < numSteps; step++) {
                try (NDManager stepManager = manager.newSubManager()) {
                    // Random xStart in [-1, 1]
                    NDArray xStart = stepManager.randomUniform(-1f, 1f, xShape);

                    // Random timesteps in [0, numTimesteps)
                    int[] tArr = new int[batchSize];
                    for (int i = 0; i < batchSize; i++) {
                        tArr[i] = (int) (Math.random() * numTimesteps);
                    }
                    NDArray t = stepManager.create(tArr);

                    // Forward + loss + backward
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray perSampleLoss = diffusion.trainingLosses(denoiseFn, xStart, t, null);
                        NDArray loss = perSampleLoss.mean();
                        gc.backward(loss);
                        losses[step] = loss.getFloat();
                    }

                    // Gradient clipping (global norm)
                    float totalNormSq = 0;
                    for (var pair : model.getParameters()) {
                        if (pair.getValue().requiresGradient()
                                && pair.getValue().getArray().hasGradient()) {
                            NDArray grad = pair.getValue().getArray().getGradient();
                            totalNormSq += grad.square().sum().getFloat();
                        }
                    }
                    float totalNorm = (float) Math.sqrt(totalNormSq);
                    float maxNorm = 1.0f;
                    if (totalNorm > maxNorm) {
                        float scale = maxNorm / (totalNorm + 1e-6f);
                        for (var pair : model.getParameters()) {
                            if (pair.getValue().requiresGradient()
                                    && pair.getValue().getArray().hasGradient()) {
                                pair.getValue().getArray().getGradient().muli(scale);
                            }
                        }
                    }

                    // Optimizer step
                    for (var pair : model.getParameters()) {
                        if (pair.getValue().requiresGradient()
                                && pair.getValue().getArray().hasGradient()) {
                            NDArray weight = pair.getValue().getArray();
                            NDArray grad = weight.getGradient();
                            optimizer.update(pair.getKey(), weight, grad);
                        }
                    }
                }
            }

            // --- Assertions ---
            for (int i = 0; i < numSteps; i++) {
                assertTrue(Float.isFinite(losses[i]),
                        "Loss at step " + i + " should be finite, got " + losses[i]);
            }
        }
    }
}
