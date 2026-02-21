package com.technodrome.diffusion.training;

import ai.djl.training.tracker.Tracker;

/**
 * Linear warmup learning rate scheduler.
 * Ported from utils.py: get_warmed_up_lr().
 *
 * LR ramps linearly from 0 to maxLR over warmupSteps, then stays at maxLR.
 */
public class LRScheduler {

    /**
     * Create a DJL Tracker that implements linear warmup.
     *
     * @param maxLr       target learning rate
     * @param warmupSteps number of warmup steps (0 = no warmup)
     * @return DJL Tracker for use with optimizers
     */
    public static Tracker createWarmupTracker(float maxLr, int warmupSteps) {
        if (warmupSteps <= 0) {
            return Tracker.fixed(maxLr);
        }
        // Custom linear warmup: ramp from 0 to maxLr over warmupSteps, then constant
        final int steps = warmupSteps;
        final float lr = maxLr;
        return numUpdate -> {
            if (numUpdate < steps) {
                return lr * numUpdate / steps;
            }
            return lr;
        };
    }

    /**
     * Compute warmed-up LR manually (for logging/debugging).
     */
    public static float getWarmedUpLr(float maxLr, int warmupSteps, int globalStep) {
        if (warmupSteps == 0) {
            return maxLr;
        }
        return maxLr * Math.min((float) globalStep / warmupSteps, 1.0f);
    }
}
