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
import ai.djl.training.tracker.Tracker;
import ai.djl.util.cuda.CudaUtils;
import com.technodrome.diffusion.util.GpuMemoryUtils;
import com.technodrome.diffusion.util.ImageUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
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
        public int saveInterval = 5000;
        public int keepCheckpointMax = 2;
        public int sampleBatchSize = 16;
        public Path outputDir = Path.of("output");
        public boolean randflip = true;
        public int numTimesteps = 1000;
        public int heapDumpAtStep = 0; // Set >0 to dump heap at that step (0 = disabled)
        public Device device = null; // null = use engine default
    }

    private final Block model;
    private final ManualAdam optimizer;
    private final EMAHelper ema;
    private final CheckpointManager checkpointManager;
    private final Config config;
    private final Random random = new Random();
    private final NDManager manager;
    private final ParameterStore ps;
    private int globalStep = 0;

    public DiffusionTrainer(Block model, Config config) {
        this.model = model;
        this.config = config;

        // Single NDManager for the entire training lifecycle
        Device resolvedDevice = config.device != null
                ? config.device
                : ai.djl.engine.Engine.getInstance().defaultDevice();
        this.manager = NDManager.newBaseManager(resolvedDevice);
        this.ps = new ParameterStore(manager, false);

        // Manual Adam optimizer with warmup LR schedule
        // (DJL's built-in Adam leaks temporary NDArrays on the parent manager)
        Tracker lrTracker = LRScheduler.createWarmupTracker(config.lr, config.warmup);
        this.optimizer = new ManualAdam(lrTracker, 1e-8f);

        this.ema = new EMAHelper(model, config.emaDecay);
        this.checkpointManager = new CheckpointManager(
                config.outputDir.resolve("checkpoints"), config.keepCheckpointMax);
    }

    public ParameterStore getParameterStore() {
        return ps;
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
        Device device = manager.getDevice();
        logger.info("Training on device: {}", device);

        // Initialize model
        Shape inputShape = new Shape(config.sampleBatchSize, imageShape.get(0),
                imageShape.get(1), imageShape.get(2));
        model.initialize(manager, DataType.FLOAT32, inputShape, new Shape(config.sampleBatchSize));
        logGpuMemory("after model init", device);

        ema.register(manager);
        logGpuMemory("after EMA register", device);

        // Try to load latest checkpoint
        try {
            int loadedStep = checkpointManager.loadLatest(model, manager);
            if (loadedStep >= 0) {
                globalStep = loadedStep;
                // loadParameters() replaces NDArrays, losing native requires_grad flag.
                // Re-enable gradient tracking on all trainable parameters.
                for (var pair : model.getParameters()) {
                    if (pair.getValue().requiresGradient()) {
                        pair.getValue().getArray().setRequiresGradient(true);
                    }
                }
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
                    if (globalStep == 0) logGpuMemory("step 0 start", device);

                    // Attach xStart to stepManager so all derived intermediates
                    // get freed when stepManager closes (prevents GPU memory leak)
                    NDArray xStart = batch.getData().singletonOrThrow().toDevice(device, false);
                    xStart.attach(stepManager);
                    batch.close(); // batch data no longer needed, xStart is on stepManager

                    // Random horizontal flip
                    if (config.randflip) {
                        xStart = randomFlipLR(xStart, stepManager);
                    }

                    long batchSize = xStart.getShape().get(0);
                    if (globalStep == 0) logGpuMemory("step 0 after data load (bs=" + batchSize + ")", device);

                    // Random timesteps
                    int[] tArr = new int[(int) batchSize];
                    for (int i = 0; i < tArr.length; i++) {
                        tArr[i] = random.nextInt(config.numTimesteps);
                    }
                    NDArray t = stepManager.create(tArr).toDevice(device, false);

                    // Phase-level resource tracking (every 100 steps)
                    long r0 = 0;
                    boolean phaseLog = (globalStep % 100 == 0);
                    if (phaseLog) r0 = countManagerResources(manager)[0];

                    // Forward pass + loss
                    float lossValue;
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        // DJL's GradientCollector does NOT zero gradients — PyTorch accumulates
                        // them across backward() calls. Must zero before each step.
                        zeroGradients();

                        NDArray losses = lossFn.apply(xStart, t);
                        if (globalStep == 0) logGpuMemory("step 0 after forward", device);
                        NDArray loss = losses.mean();
                        gc.backward(loss);
                        lossValue = loss.toFloatArray()[0];
                        if (globalStep == 0) logGpuMemory("step 0 after backward", device);
                    }
                    if (globalStep == 0) logGpuMemory("step 0 after gc close", device);
                    if (phaseLog) {
                        long r1 = countManagerResources(manager)[0];
                        logger.info("[phase] Step {}: after fwd+bwd+gc: parent +{}", globalStep, r1 - r0);
                        r0 = r1;
                    }

                    // Gradient clipping
                    clipGradients(config.gradClip);
                    if (phaseLog) {
                        long r1 = countManagerResources(manager)[0];
                        logger.info("[phase] Step {}: after clipGrad: parent +{}", globalStep, r1 - r0);
                        r0 = r1;
                    }

                    // Optimizer step (ManualAdam closes all temps explicitly)
                    optimizer.step(model);
                    if (phaseLog) {
                        long r1 = countManagerResources(manager)[0];
                        logger.info("[phase] Step {}: after optimizer: parent +{}", globalStep, r1 - r0);
                        r0 = r1;
                    }

                    // EMA update
                    ema.update();
                    if (phaseLog) {
                        long r1 = countManagerResources(manager)[0];
                        logger.info("[phase] Step {}: after EMA: parent +{}", globalStep, r1 - r0);
                        r0 = r1;
                    }

                    globalStep++;

                    // Logging
                    if (globalStep % config.logInterval == 0) {
                        logger.info("Step {}: loss = {}", globalStep, String.format("%.6f", lossValue));
                    }

                    // Sample generation
                    if (globalStep % config.sampleInterval == 0) {
                        generateAndSaveSamples(manager, sampleFn, imageShape, samplesDir);
                    }

                    // Checkpoint
                    if (globalStep % config.saveInterval == 0) {
                        saveCheckpoint(globalStep);
                    }
                    if (globalStep == 1) logGpuMemory("step 1 end (before stepMgr close)", device);
                }
                // Periodically run Java GC to free orphaned NDArray finalizers
                if (globalStep % 10 == 0) {
                    System.gc();
                }
                if (globalStep == 1) logGpuMemory("step 1 (after stepMgr close + gc)", device);

                // Detailed diagnostics every 100 steps (measured AFTER stepManager close)
                if (globalStep % 100 == 0) {
                    long[] counts = countManagerResources(manager);
                    if (globalStep % 1000 == 0) {
                        System.gc();
                        Runtime rt = Runtime.getRuntime();
                        long usedMB = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
                        long maxMB = rt.maxMemory() / (1024 * 1024);
                        logger.info("[Java heap] Step {}: {}MB used / {}MB max | " +
                                "manager: resources={}, tempResources={}, children={}",
                                globalStep, usedMB, maxMB, counts[0], counts[1], counts[2]);
                        logResourceTypes(manager);
                    } else {
                        logger.info("[manager] Step {}: resources={}, tempResources={}, children={}",
                                globalStep, counts[0], counts[1], counts[2]);
                    }
                }


                // On-demand heap dump for leak analysis
                if (config.heapDumpAtStep > 0 && globalStep == config.heapDumpAtStep) {
                    dumpHeap(config.outputDir.resolve(
                            String.format("heap_step%d.hprof", globalStep)).toString());
                }
            }
        }

        // Final save
        saveCheckpoint(globalStep);
        optimizer.close();
        manager.close();
        logger.info("Training complete at step {}", globalStep);
    }

    private void generateAndSaveSamples(NDManager manager,
                                         BiFunction<NDManager, Shape, NDArray> sampleFn,
                                         Shape imageShape, Path samplesDir) {
        logger.info("Generating samples at step {}...", globalStep);
        // Free cached training memory before the 1000-step sampling loop
        GpuMemoryUtils.cudaEmptyCache();
        try (NDManager sampleManager = manager.newSubManager()) {
            Shape sampleShape = new Shape(config.sampleBatchSize,
                    imageShape.get(0), imageShape.get(1), imageShape.get(2));

            // Use EMA parameters for sampling (swap in, sample, swap back)
            ema.swapWithEma();
            NDArray samples = sampleFn.apply(sampleManager, sampleShape);
            ema.swapWithEma();

            // Save tiled image
            String filename = String.format("samples_%d.png", globalStep);
            ImageUtils.saveTiledImages(samplesDir.resolve(filename), samples);
            logger.info("Samples saved to {}", filename);
        } catch (Exception e) {
            logger.error("Failed to generate samples", e);
        }
        // Release PyTorch's cached CUDA memory after the heavy sampling loop
        GpuMemoryUtils.cudaEmptyCache();
    }

    /**
     * Save checkpoint with leak containment.
     * DJL's saveParameters() calls toByteBuffer() on each GPU parameter, which leaks
     * a PtNDManager per param via JniUtils.getByteBuffer() → toDevice(cpu) → newSubManager().
     * We temporarily move params to a scoped manager so leaked sub-managers go there.
     */
    private void saveCheckpoint(int step) throws java.io.IOException {
        try (NDManager saveMgr = manager.newSubManager()) {
            // Move params to scoped manager so toByteBuffer() leaks go there
            for (var pair : model.getParameters()) {
                pair.getValue().getArray().attach(saveMgr);
            }
            try {
                checkpointManager.save(model, step);
            } finally {
                // Always move params back to parent, even on error
                for (var pair : model.getParameters()) {
                    pair.getValue().getArray().attach(manager);
                }
            }
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
        // Compute global gradient norm.
        // IMPORTANT: Both getFloat() and toFloatArray() on GPU tensors leak sub-managers.
        // JniUtils.getByteBuffer() calls toDevice(cpu) which creates a sub-manager via
        // JniUtils.to() that is never closed. We use a scoped clipMgr: attach `s` to it
        // before extracting the float so leaked sub-managers go on clipMgr, not the parent.
        float totalNormSq = 0;
        try (NDManager clipMgr = manager.newSubManager()) {
            for (var pair : model.getParameters()) {
                if (pair.getValue().requiresGradient() && pair.getValue().getArray().hasGradient()) {
                    try (NDArray grad = pair.getValue().getArray().getGradient()) {
                        try (NDArray sq = grad.square();
                             NDArray s = sq.sum()) {
                            s.attach(clipMgr);
                            totalNormSq += s.toFloatArray()[0];
                        }
                    }
                }
            }
        }
        float totalNorm = (float) Math.sqrt(totalNormSq);

        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            try (NDArray scaleScalar = manager.create(scale)) {
                for (var pair : model.getParameters()) {
                    if (pair.getValue().requiresGradient() && pair.getValue().getArray().hasGradient()) {
                        try (NDArray grad = pair.getValue().getArray().getGradient()) {
                            grad.muli(scaleScalar);
                        }
                    }
                }
            }
        }
    }

    /**
     * Zero all parameter gradients before each backward pass.
     * DJL's PtGradientCollector does NOT zero gradients — PyTorch's backward()
     * accumulates into existing .grad tensors. Without zeroing, gradients from
     * all prior steps accumulate, causing the optimizer to use a stale average
     * direction and the loss to plateau.
     *
     * DJL's built-in zeroGradients() leaks 2 getGradient() wrappers per parameter
     * per call (never closed). This version properly closes them via try-with-resources.
     */
    private void zeroGradients() {
        for (var pair : model.getParameters()) {
            if (pair.getValue().requiresGradient() && pair.getValue().getArray().hasGradient()) {
                try (NDArray grad = pair.getValue().getArray().getGradient()) {
                    grad.subi(grad); // zero in-place: grad = grad - grad = 0
                }
            }
        }
    }

    public int getGlobalStep() {
        return globalStep;
    }

    private void logGpuMemory(String label, Device device) {
        return;
//        if (device.isGpu()) {
//            try {
//                MemoryUsage mem = CudaUtils.getGpuMemory(device);
//                long usedMB = mem.getCommitted() / (1024 * 1024);
//                long totalMB = mem.getMax() / (1024 * 1024);
//                long freeMB = totalMB - usedMB;
//                logger.info("[GPU mem] {}: {}MB used / {}MB total ({}MB free)",
//                        label, usedMB, totalMB, freeMB);
//            } catch (Exception e) {
//                // ignore if memory query fails
//            }
//        }
    }

    /**
     * Count managed resources on an NDManager using reflection.
     * Returns [resources_count, tempResources_count, children_count].
     */
    private long[] countManagerResources(NDManager mgr) {
        long[] counts = {-1, -1, -1};
        try {
            Class<?> clazz = mgr.getClass();
            // Walk up class hierarchy to find BaseNDManager fields
            while (clazz != null) {
                try {
                    java.lang.reflect.Field f = clazz.getDeclaredField("resources");
                    f.setAccessible(true);
                    Object map = f.get(mgr);
                    if (map instanceof java.util.Map) {
                        counts[0] = ((java.util.Map<?, ?>) map).size();
                    }
                } catch (NoSuchFieldException ignored) {}
                try {
                    java.lang.reflect.Field f = clazz.getDeclaredField("tempResources");
                    f.setAccessible(true);
                    Object map = f.get(mgr);
                    if (map instanceof java.util.Map) {
                        counts[1] = ((java.util.Map<?, ?>) map).size();
                    }
                } catch (NoSuchFieldException ignored) {}
                try {
                    java.lang.reflect.Field f = clazz.getDeclaredField("children");
                    f.setAccessible(true);
                    Object list = f.get(mgr);
                    if (list instanceof java.util.Collection) {
                        counts[2] = ((java.util.Collection<?>) list).size();
                    }
                } catch (NoSuchFieldException ignored) {}
                clazz = clazz.getSuperclass();
            }
        } catch (Exception e) {
            logger.debug("Failed to count manager resources: {}", e.getMessage());
        }
        return counts;
    }

    /**
     * Log a breakdown of object types in the parent manager's resources map.
     * Helps identify what's accumulating (NDArrays, sub-managers, etc.).
     */
    private void logResourceTypes(NDManager mgr) {
        try {
            Class<?> clazz = mgr.getClass();
            while (clazz != null) {
                try {
                    java.lang.reflect.Field f = clazz.getDeclaredField("resources");
                    f.setAccessible(true);
                    Object map = f.get(mgr);
                    if (map instanceof java.util.Map<?, ?> resources) {
                        java.util.Map<String, Integer> typeCounts = new java.util.TreeMap<>();
                        for (Object val : resources.values()) {
                            String typeName = val.getClass().getSimpleName();
                            typeCounts.merge(typeName, 1, Integer::sum);
                        }
                        logger.info("[manager resources breakdown] {}", typeCounts);
                        return;
                    }
                } catch (NoSuchFieldException ignored) {}
                clazz = clazz.getSuperclass();
            }
        } catch (Exception e) {
            logger.debug("Failed to log resource types: {}", e.getMessage());
        }
    }

    /** Dump live heap to an .hprof file for analysis in Eclipse MAT / IntelliJ. */
    private void dumpHeap(String filePath) {
        try {
            System.gc();
            logger.info("Dumping heap to {} ...", filePath);
            java.nio.file.Files.createDirectories(Path.of(filePath).getParent());
            var bean = ManagementFactory.getPlatformMXBean(
                    com.sun.management.HotSpotDiagnosticMXBean.class);
            bean.dumpHeap(filePath, true); // true = only live objects
            logger.info("Heap dump complete: {}", filePath);
        } catch (Exception e) {
            logger.error("Failed to dump heap", e);
        }
    }

    // Inner static class for Engine access
    private static class Engine {
        static ai.djl.engine.Engine getInstance() {
            return ai.djl.engine.Engine.getInstance();
        }
    }
}
