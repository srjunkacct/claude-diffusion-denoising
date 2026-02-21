package com.technodrome.diffusion.diffusion;

/**
 * Beta noise schedule computation.
 * Ported from diffusion_utils.py: get_beta_schedule(), _warmup_beta().
 * Pure math — all computations in double precision.
 */
public final class BetaSchedule {

    private BetaSchedule() {}

    /**
     * Compute the beta schedule for the diffusion process.
     *
     * @param schedule     one of: "linear", "quad", "warmup10", "warmup50", "const", "jsd"
     * @param betaStart    starting beta value (e.g. 0.0001)
     * @param betaEnd      ending beta value (e.g. 0.02)
     * @param numTimesteps number of diffusion timesteps (e.g. 1000)
     * @return array of betas in float64
     */
    public static double[] getBetaSchedule(String schedule, double betaStart, double betaEnd,
                                           int numTimesteps) {
        double[] betas;
        switch (schedule) {
            case "linear":
                betas = linspace(betaStart, betaEnd, numTimesteps);
                break;
            case "quad":
                betas = linspace(Math.sqrt(betaStart), Math.sqrt(betaEnd), numTimesteps);
                for (int i = 0; i < betas.length; i++) {
                    betas[i] = betas[i] * betas[i];
                }
                break;
            case "warmup10":
                betas = warmupBeta(betaStart, betaEnd, numTimesteps, 0.1);
                break;
            case "warmup50":
                betas = warmupBeta(betaStart, betaEnd, numTimesteps, 0.5);
                break;
            case "const":
                betas = new double[numTimesteps];
                java.util.Arrays.fill(betas, betaEnd);
                break;
            case "jsd":
                // 1/T, 1/(T-1), ..., 1/1
                betas = new double[numTimesteps];
                for (int i = 0; i < numTimesteps; i++) {
                    betas[i] = 1.0 / (numTimesteps - i);
                }
                break;
            default:
                throw new UnsupportedOperationException("Unknown beta schedule: " + schedule);
        }
        return betas;
    }

    private static double[] warmupBeta(double betaStart, double betaEnd,
                                        int numTimesteps, double warmupFrac) {
        double[] betas = new double[numTimesteps];
        java.util.Arrays.fill(betas, betaEnd);
        int warmupTime = (int) (numTimesteps * warmupFrac);
        double[] warmup = linspace(betaStart, betaEnd, warmupTime);
        System.arraycopy(warmup, 0, betas, 0, warmupTime);
        return betas;
    }

    /** Linearly spaced values from start to end (inclusive), like numpy.linspace. */
    static double[] linspace(double start, double end, int num) {
        double[] result = new double[num];
        if (num == 1) {
            result[0] = start;
            return result;
        }
        for (int i = 0; i < num; i++) {
            result[i] = start + (end - start) * i / (num - 1);
        }
        return result;
    }
}
