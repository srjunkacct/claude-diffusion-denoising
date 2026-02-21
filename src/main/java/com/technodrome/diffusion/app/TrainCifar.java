package com.technodrome.diffusion.app;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Dataset;
import com.technodrome.diffusion.dataset.CifarDataset;
import com.technodrome.diffusion.diffusion.BetaSchedule;
import com.technodrome.diffusion.diffusion.GaussianDiffusion2;
import com.technodrome.diffusion.model.UNetBlock;
import com.technodrome.diffusion.training.DiffusionTrainer;
import com.technodrome.diffusion.util.SeedUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.nio.file.Path;
import java.util.function.BiFunction;

/**
 * CIFAR-10 unconditional diffusion training.
 * Ported from run_cifar.py.
 *
 * Model: unet2d16b2 (35.7M params)
 * Diffusion: GaussianDiffusion2 (eps, fixedlarge, mse)
 * Config: ch=128, ch_mult=(1,2,2,2), 32x32, bs=128, lr=2e-4, dropout=0.1
 */
@Command(name = "train-cifar", description = "Train DDPM on CIFAR-10")
public class TrainCifar implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(TrainCifar.class);

    @Option(names = "--lr", defaultValue = "2e-4", description = "Learning rate")
    double lr;

    @Option(names = "--batch-size", defaultValue = "128", description = "Batch size")
    int batchSize;

    @Option(names = "--warmup", defaultValue = "5000", description = "LR warmup steps")
    int warmup;

    @Option(names = "--grad-clip", defaultValue = "1.0", description = "Gradient clipping norm")
    float gradClip;

    @Option(names = "--dropout", defaultValue = "0.1", description = "Dropout rate")
    float dropout;

    @Option(names = "--timesteps", defaultValue = "1000", description = "Number of diffusion timesteps")
    int timesteps;

    @Option(names = "--beta-start", defaultValue = "0.0001", description = "Beta schedule start")
    double betaStart;

    @Option(names = "--beta-end", defaultValue = "0.02", description = "Beta schedule end")
    double betaEnd;

    @Option(names = "--beta-schedule", defaultValue = "linear", description = "Beta schedule type")
    String betaSchedule;

    @Option(names = "--model-mean-type", defaultValue = "eps", description = "Model mean type: eps, xstart, xprev")
    String modelMeanType;

    @Option(names = "--model-var-type", defaultValue = "fixedlarge", description = "Model variance type: fixedlarge, fixedsmall, learned")
    String modelVarType;

    @Option(names = "--loss-type", defaultValue = "mse", description = "Loss type: mse, kl")
    String lossType;

    @Option(names = "--total-steps", defaultValue = "800000", description = "Total training steps")
    int totalSteps;

    @Option(names = "--output-dir", defaultValue = "output/cifar", description = "Output directory")
    Path outputDir;

    @Option(names = "--seed", defaultValue = "42", description = "Random seed")
    long seed;

    @Option(names = "--randflip", defaultValue = "true", description = "Random horizontal flip augmentation")
    boolean randflip;

    @Override
    public void run() {
        try {
            SeedUtils.seedAll(seed);
            logger.info("Training CIFAR-10 DDPM");
            logger.info("Config: lr={}, bs={}, dropout={}, timesteps={}, mean={}, var={}, loss={}",
                    lr, batchSize, dropout, timesteps, modelMeanType, modelVarType, lossType);

            // Beta schedule
            double[] betas = BetaSchedule.getBetaSchedule(betaSchedule, betaStart, betaEnd, timesteps);

            // Diffusion process
            GaussianDiffusion2 diffusion = new GaussianDiffusion2(
                    betas, modelMeanType, modelVarType, lossType);

            // U-Net model: ch=128, ch_mult=(1,2,2,2), attn at 16x16
            int ch = 128;
            int[] chMult = {1, 2, 2, 2};
            int[] attnRes = {16};
            int outCh = "learned".equals(modelVarType) ? 6 : 3;

            Block model = new UNetBlock(ch, outCh, chMult, 2, attnRes, dropout, true, 32);

            // Dataset
            Dataset dataset = CifarDataset.getTrainDataset(batchSize, true);
            dataset.prepare();

            // Denoise function: (x, t) -> model output
            NDManager baseManager = NDManager.newBaseManager();
            ParameterStore ps = new ParameterStore(baseManager, false);
            model.initialize(baseManager, DataType.FLOAT32,
                    new Shape(batchSize, 3, 32, 32), new Shape(batchSize));

            BiFunction<NDArray, NDArray, NDArray> denoiseFn = (x, t) ->
                    model.forward(ps, new NDList(x, t), true).singletonOrThrow();

            // Loss function
            BiFunction<NDArray, NDArray, NDArray> lossFn = (xStart, t) ->
                    diffusion.trainingLosses(denoiseFn, xStart, t, null);

            // Sample function
            BiFunction<NDManager, Shape, NDArray> sampleFn = (mgr, shape) -> {
                BiFunction<NDArray, NDArray, NDArray> evalDenoiseFn = (x, t) ->
                        model.forward(ps, new NDList(x, t), false).singletonOrThrow();
                return diffusion.pSampleLoop(evalDenoiseFn, mgr, shape);
            };

            // Trainer
            DiffusionTrainer.Config trainerConfig = new DiffusionTrainer.Config();
            trainerConfig.lr = (float) lr;
            trainerConfig.warmup = warmup;
            trainerConfig.gradClip = gradClip;
            trainerConfig.outputDir = outputDir;
            trainerConfig.randflip = randflip;
            trainerConfig.sampleBatchSize = 16;

            DiffusionTrainer trainer = new DiffusionTrainer(model, trainerConfig);
            trainer.train(dataset, lossFn, sampleFn, totalSteps, new Shape(3, 32, 32));

        } catch (Exception e) {
            logger.error("Training failed", e);
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new TrainCifar()).execute(args);
        System.exit(exitCode);
    }
}
