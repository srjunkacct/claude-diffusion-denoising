package com.technodrome.diffusion.app;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Dataset;
import com.technodrome.diffusion.dataset.LSUNDataset;
import com.technodrome.diffusion.diffusion.BetaSchedule;
import com.technodrome.diffusion.diffusion.GaussianDiffusion;
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
 * LSUN (church, bedroom, cat) 256x256 unconditional diffusion training.
 * Ported from run_lsun.py.
 *
 * Model: unet2d16b2c112244 (114M params)
 * Diffusion: GaussianDiffusion (noisepred)
 * Config: ch=128, ch_mult=(1,1,2,2,4,4), 256x256, bs=64, lr=2e-5
 */
@Command(name = "train-lsun", description = "Train DDPM on LSUN 256x256")
public class TrainLSUN implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(TrainLSUN.class);

    @Option(names = "--lr", defaultValue = "2e-5", description = "Learning rate")
    double lr;

    @Option(names = "--batch-size", defaultValue = "64", description = "Batch size")
    int batchSize;

    @Option(names = "--warmup", defaultValue = "5000", description = "LR warmup steps")
    int warmup;

    @Option(names = "--grad-clip", defaultValue = "1.0", description = "Gradient clipping norm")
    float gradClip;

    @Option(names = "--dropout", defaultValue = "0.0", description = "Dropout rate")
    float dropout;

    @Option(names = "--timesteps", defaultValue = "1000", description = "Number of diffusion timesteps")
    int timesteps;

    @Option(names = "--beta-start", defaultValue = "0.0001", description = "Beta schedule start")
    double betaStart;

    @Option(names = "--beta-end", defaultValue = "0.02", description = "Beta schedule end")
    double betaEnd;

    @Option(names = "--beta-schedule", defaultValue = "linear", description = "Beta schedule type")
    String betaSchedule;

    @Option(names = "--loss-type", defaultValue = "noisepred", description = "Loss type")
    String lossType;

    @Option(names = "--total-steps", defaultValue = "500000", description = "Total training steps")
    int totalSteps;

    @Option(names = "--data-dir", required = true, description = "Directory containing LSUN images")
    Path dataDir;

    @Option(names = "--output-dir", defaultValue = "output/lsun", description = "Output directory")
    Path outputDir;

    @Option(names = "--seed", defaultValue = "42", description = "Random seed")
    long seed;

    @Option(names = "--randflip", defaultValue = "true", description = "Random horizontal flip (set false for cats)")
    boolean randflip;

    @Option(names = "--max-images", defaultValue = "0", description = "Max images to load (0=all)")
    int maxImages;

    @Override
    public void run() {
        try {
            SeedUtils.seedAll(seed);
            logger.info("Training LSUN 256x256 DDPM");
            logger.info("Config: lr={}, bs={}, dropout={}, timesteps={}, loss={}, randflip={}",
                    lr, batchSize, dropout, timesteps, lossType, randflip);

            // Beta schedule
            double[] betas = BetaSchedule.getBetaSchedule(betaSchedule, betaStart, betaEnd, timesteps);

            // Diffusion process
            NDManager baseManager = NDManager.newBaseManager();
            GaussianDiffusion diffusion = new GaussianDiffusion(baseManager, betas, lossType);

            // U-Net model: ch=128, ch_mult=(1,1,2,2,4,4), attn at 16x16
            int ch = 128;
            int[] chMult = {1, 1, 2, 2, 4, 4};
            int[] attnRes = {16};
            Block model = new UNetBlock(ch, 3, chMult, 2, attnRes, dropout, true, 256);

            // Dataset
            Dataset dataset = LSUNDataset.getTrainDataset(dataDir, batchSize, 256, maxImages, true);
            dataset.prepare();

            // Initialize model
            ParameterStore ps = new ParameterStore(baseManager, false);
            model.initialize(baseManager, DataType.FLOAT32,
                    new Shape(batchSize, 3, 256, 256), new Shape(batchSize));

            BiFunction<NDArray, NDArray, NDArray> denoiseFn = (x, t) ->
                    model.forward(ps, new NDList(x, t), true).singletonOrThrow();

            BiFunction<NDArray, NDArray, NDArray> lossFn = (xStart, t) ->
                    diffusion.pLosses(denoiseFn, xStart, t, null);

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
            trainerConfig.numTimesteps = timesteps;
            trainerConfig.sampleBatchSize = 4;

            DiffusionTrainer trainer = new DiffusionTrainer(model, trainerConfig);
            trainer.train(dataset, lossFn, sampleFn, totalSteps, new Shape(3, 256, 256));

        } catch (Exception e) {
            logger.error("Training failed", e);
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new TrainLSUN()).execute(args);
        System.exit(exitCode);
    }
}
