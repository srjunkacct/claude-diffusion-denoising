package com.technodrome.diffusion.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;

/**
 * Statistical math functions for diffusion models.
 * Ported from diffusion_utils.py: normal_kl() and utils.py: discretized_gaussian_log_likelihood,
 * approx_standard_normal_cdf.
 */
public final class MathUtils {

    private MathUtils() {}

    /**
     * KL divergence between two normal distributions parameterized by mean and log-variance.
     * KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)))
     */
    public static NDArray normalKl(NDArray mean1, NDArray logvar1, NDArray mean2, NDArray logvar2) {
        // 0.5 * (-1 + logvar2 - logvar1 + exp(logvar1 - logvar2) + (mean1 - mean2)^2 * exp(-logvar2))
        NDArray diff = mean1.sub(mean2);
        return logvar2.sub(logvar1)
                .add(logvar1.sub(logvar2).exp())
                .add(diff.square().mul(logvar2.neg().exp()))
                .sub(1.0)
                .mul(0.5);
    }

    /**
     * Approximate the standard normal CDF using tanh.
     * From: A Handy Approximation for the Error Function and its Inverse (Sergei Winitzki, 2008)
     */
    public static NDArray approxStandardNormalCdf(NDArray x) {
        double sqrtTwoOverPi = Math.sqrt(2.0 / Math.PI);
        // 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return x.add(x.pow(3).mul(0.044715))
                .mul(sqrtTwoOverPi)
                .tanh()
                .add(1.0)
                .mul(0.5);
    }

    /**
     * Log-likelihood for discretized Gaussian.
     * Assumes data is integers [0, 255] rescaled to [-1, 1].
     *
     * @param x         data tensor
     * @param means     predicted means
     * @param logScales predicted log standard deviations
     * @return log probability per element
     */
    public static NDArray discretizedGaussianLogLikelihood(NDArray x, NDArray means,
                                                           NDArray logScales) {
        NDArray centeredX = x.sub(means);
        NDArray invStdv = logScales.neg().exp();
        NDArray plusIn = invStdv.mul(centeredX.add(1.0 / 255.0));
        NDArray cdfPlus = approxStandardNormalCdf(plusIn);
        NDArray minIn = invStdv.mul(centeredX.sub(1.0 / 255.0));
        NDArray cdfMin = approxStandardNormalCdf(minIn);

        NDArray logCdfPlus = cdfPlus.maximum(1e-12).log();
        NDArray logOneMinusCdfMin = cdfMin.neg().add(1.0).maximum(1e-12).log();
        NDArray cdfDelta = cdfPlus.sub(cdfMin);

        // x < -0.999 -> log_cdf_plus
        // x > 0.999 -> log_one_minus_cdf_min
        // else -> log(max(cdf_delta, 1e-12))
        NDArray logProbs = cdfDelta.maximum(1e-12).log();
        logProbs = NDArrays.where(x.lt(-0.999), logCdfPlus, logProbs);
        logProbs = NDArrays.where(x.gt(0.999), logOneMinusCdfMin, logProbs);

        return logProbs;
    }
}
