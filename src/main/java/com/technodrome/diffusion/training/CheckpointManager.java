package com.technodrome.diffusion.training;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Checkpoint save/load wrapper for DJL models.
 * Saves model parameters and training state (step count) to disk.
 */
public class CheckpointManager {

    private static final Logger logger = LoggerFactory.getLogger(CheckpointManager.class);

    private final Path checkpointDir;
    private final int keepMax;

    /**
     * @param checkpointDir directory for storing checkpoints
     * @param keepMax       maximum number of checkpoints to keep (0 = unlimited)
     */
    public CheckpointManager(Path checkpointDir, int keepMax) {
        this.checkpointDir = checkpointDir;
        this.keepMax = keepMax;
    }

    /** Save model parameters and current step. */
    public void save(Block model, int step) throws IOException {
        Path stepDir = checkpointDir.resolve("step-" + step);
        Files.createDirectories(stepDir);

        // Save block parameters
        try (DataOutputStream paramOut = new DataOutputStream(
                Files.newOutputStream(stepDir.resolve("model.params")))) {
            model.saveParameters(paramOut);
        }

        // Save step number
        try (DataOutputStream dos = new DataOutputStream(
                Files.newOutputStream(stepDir.resolve("state.bin")))) {
            dos.writeInt(step);
        }

        logger.info("Checkpoint saved at step {} to {}", step, stepDir);
        cleanOldCheckpoints();
    }

    /** Load model parameters. Returns the saved step number. */
    public int load(Block model, NDManager manager, int step) throws IOException, MalformedModelException {
        Path stepDir = checkpointDir.resolve("step-" + step);
        try (DataInputStream paramIn = new DataInputStream(
                Files.newInputStream(stepDir.resolve("model.params")))) {
            model.loadParameters(manager, paramIn);
        }
        logger.info("Checkpoint loaded from step {} at {}", step, stepDir);
        return step;
    }

    /** Load the latest checkpoint. Returns the step number, or -1 if none found. */
    public int loadLatest(Block model, NDManager manager) throws IOException, MalformedModelException {
        int latestStep = findLatestStep();
        if (latestStep < 0) {
            logger.info("No checkpoint found in {}", checkpointDir);
            return -1;
        }
        return load(model, manager, latestStep);
    }

    /** Find the latest checkpoint step number, or -1 if none. */
    public int findLatestStep() throws IOException {
        if (!Files.isDirectory(checkpointDir)) {
            return -1;
        }
        int latest = -1;
        try (var stream = Files.list(checkpointDir)) {
            for (Path p : (Iterable<Path>) stream::iterator) {
                String name = p.getFileName().toString();
                if (name.startsWith("step-")) {
                    try {
                        int step = Integer.parseInt(name.substring(5));
                        latest = Math.max(latest, step);
                    } catch (NumberFormatException ignored) {}
                }
            }
        }
        return latest;
    }

    private void cleanOldCheckpoints() throws IOException {
        if (keepMax <= 0 || !Files.isDirectory(checkpointDir)) return;

        java.util.List<Integer> steps = new java.util.ArrayList<>();
        try (var stream = Files.list(checkpointDir)) {
            for (Path p : (Iterable<Path>) stream::iterator) {
                String name = p.getFileName().toString();
                if (name.startsWith("step-")) {
                    try {
                        steps.add(Integer.parseInt(name.substring(5)));
                    } catch (NumberFormatException ignored) {}
                }
            }
        }

        if (steps.size() <= keepMax) return;

        steps.sort(java.util.Comparator.naturalOrder());
        for (int i = 0; i < steps.size() - keepMax; i++) {
            Path oldDir = checkpointDir.resolve("step-" + steps.get(i));
            deleteRecursive(oldDir);
            logger.info("Removed old checkpoint: {}", oldDir);
        }
    }

    private static void deleteRecursive(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (var stream = Files.list(path)) {
                for (Path child : (Iterable<Path>) stream::iterator) {
                    deleteRecursive(child);
                }
            }
        }
        Files.deleteIfExists(path);
    }
}
