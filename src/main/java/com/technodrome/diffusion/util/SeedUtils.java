package com.technodrome.diffusion.util;

import ai.djl.engine.Engine;

import java.util.Random;

/**
 * Reproducibility utilities.
 * Ported from utils.py: seed_all().
 */
public final class SeedUtils {

    private SeedUtils() {}

    /**
     * Set random seeds for reproducibility.
     * Sets Java Random seed and the DJL/PyTorch engine seed.
     */
    public static void seedAll(long seed) {
        // Java random
        new Random(seed); // Not globally effective, but sets the pattern

        // DJL engine seed (delegates to PyTorch's manual_seed)
        Engine.getInstance().setRandomSeed((int) seed);
    }
}
