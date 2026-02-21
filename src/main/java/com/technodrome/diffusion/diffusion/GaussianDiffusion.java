package com.technodrome.diffusion.diffusion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.technodrome.diffusion.nn.NNUtils;

import java.util.function.BiFunction;

/**
 * Gaussian diffusion process utilities (simple version).
 * Ported from diffusion_utils.py: GaussianDiffusion.
 *
 * Supports 'noisepred' loss type (noise prediction MSE).
 * Used by CelebA-HQ and LSUN training scripts.
 *
 * All coefficient arrays are stored as NDArrays on the given manager's device.
 * tf.while_loop -> plain Java for-loop (eager execution with DJL).
 */
public class GaussianDiffusion {

    private final String lossType;
    public final int numTimesteps;

    // Precomputed coefficients as NDArrays
    private final NDArray betas;
    private final NDArray alphasCumprod;
    private final NDArray alphasCumprodPrev;
    private final NDArray sqrtAlphasCumprod;
    private final NDArray sqrtOneMinusAlphasCumprod;
    private final NDArray logOneMinusAlphasCumprod;
    private final NDArray sqrtRecipAlphasCumprod;
    private final NDArray sqrtRecipm1AlphasCumprod;
    private final NDArray posteriorVariance;
    private final NDArray posteriorLogVarianceClipped;
    private final NDArray posteriorMeanCoef1;
    private final NDArray posteriorMeanCoef2;

    /**
     * @param manager  NDManager for creating coefficient tensors
     * @param betasArr beta schedule as double array (computed in float64 for accuracy)
     * @param lossType loss type: "noisepred"
     */
    public GaussianDiffusion(NDManager manager, double[] betasArr, String lossType) {
        this.lossType = lossType;
        this.numTimesteps = betasArr.length;

        // Compute all coefficients in float64 for accuracy
        double[] alphas = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            alphas[i] = 1.0 - betasArr[i];
        }
        double[] alphasCp = cumprod(alphas);
        double[] alphasCpPrev = new double[numTimesteps];
        alphasCpPrev[0] = 1.0;
        System.arraycopy(alphasCp, 0, alphasCpPrev, 1, numTimesteps - 1);

        double[] postVar = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            postVar[i] = betasArr[i] * (1.0 - alphasCpPrev[i]) / (1.0 - alphasCp[i]);
        }
        double[] postLogVarClipped = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            postLogVarClipped[i] = Math.log(Math.max(postVar[i], 1e-20));
        }
        double[] postMeanC1 = new double[numTimesteps];
        double[] postMeanC2 = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            postMeanC1[i] = betasArr[i] * Math.sqrt(alphasCpPrev[i]) / (1.0 - alphasCp[i]);
            postMeanC2[i] = (1.0 - alphasCpPrev[i]) * Math.sqrt(alphas[i]) / (1.0 - alphasCp[i]);
        }

        // Store as float32 NDArrays
        betas = manager.create(toFloat(betasArr));
        alphasCumprod = manager.create(toFloat(alphasCp));
        alphasCumprodPrev = manager.create(toFloat(alphasCpPrev));
        sqrtAlphasCumprod = manager.create(toFloat(sqrt(alphasCp)));
        sqrtOneMinusAlphasCumprod = manager.create(toFloat(sqrt(oneMinus(alphasCp))));
        logOneMinusAlphasCumprod = manager.create(toFloat(log(oneMinus(alphasCp))));
        sqrtRecipAlphasCumprod = manager.create(toFloat(sqrt(recip(alphasCp))));
        sqrtRecipm1AlphasCumprod = manager.create(toFloat(sqrt(recipM1(alphasCp))));
        posteriorVariance = manager.create(toFloat(postVar));
        posteriorLogVarianceClipped = manager.create(toFloat(postLogVarClipped));
        posteriorMeanCoef1 = manager.create(toFloat(postMeanC1));
        posteriorMeanCoef2 = manager.create(toFloat(postMeanC2));
    }

    // --- Extract coefficient at timestep and broadcast ---
    private static NDArray extract(NDArray a, NDArray t, Shape xShape) {
        // a: [T], t: [B] -> gather a[t[i]] -> [B] -> reshape for broadcast
        long[] tLong = t.toLongArray();
        float[] aData = a.toFloatArray();
        float[] gathered = new float[tLong.length];
        for (int i = 0; i < tLong.length; i++) {
            gathered[i] = aData[(int) tLong[i]];
        }
        NDArray out = t.getManager().create(gathered);
        long[] shape = new long[xShape.dimension()];
        java.util.Arrays.fill(shape, 1);
        shape[0] = xShape.get(0);
        return out.reshape(shape);
    }

    // --- Forward process ---

    public NDArray[] qMeanVariance(NDArray xStart, NDArray t) {
        NDArray mean = extract(sqrtAlphasCumprod, t, xStart.getShape()).mul(xStart);
        NDArray variance = extract(alphasCumprod.neg().add(1), t, xStart.getShape());
        NDArray logVariance = extract(logOneMinusAlphasCumprod, t, xStart.getShape());
        return new NDArray[]{mean, variance, logVariance};
    }

    /** Diffuse the data: q(x_t | x_0). */
    public NDArray qSample(NDArray xStart, NDArray t, NDArray noise) {
        if (noise == null) {
            noise = xStart.getManager().randomNormal(xStart.getShape());
        }
        return extract(sqrtAlphasCumprod, t, xStart.getShape()).mul(xStart)
                .add(extract(sqrtOneMinusAlphasCumprod, t, xStart.getShape()).mul(noise));
    }

    /** Predict x_0 from x_t and predicted noise. */
    public NDArray predictStartFromNoise(NDArray xT, NDArray t, NDArray noise) {
        return extract(sqrtRecipAlphasCumprod, t, xT.getShape()).mul(xT)
                .sub(extract(sqrtRecipm1AlphasCumprod, t, xT.getShape()).mul(noise));
    }

    /** Posterior q(x_{t-1} | x_t, x_0). */
    public NDArray[] qPosterior(NDArray xStart, NDArray xT, NDArray t) {
        NDArray mean = extract(posteriorMeanCoef1, t, xT.getShape()).mul(xStart)
                .add(extract(posteriorMeanCoef2, t, xT.getShape()).mul(xT));
        NDArray var = extract(posteriorVariance, t, xT.getShape());
        NDArray logVarClipped = extract(posteriorLogVarianceClipped, t, xT.getShape());
        return new NDArray[]{mean, var, logVarClipped};
    }

    // --- Training loss ---

    /**
     * Compute training losses.
     *
     * @param denoiseFn function (x_noisy, t) -> predicted noise
     * @param xStart    clean images [B, C, H, W]
     * @param t         timesteps [B]
     * @param noise     optional pre-generated noise
     * @return per-sample losses [B]
     */
    public NDArray pLosses(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                           NDArray xStart, NDArray t, NDArray noise) {
        if (noise == null) {
            noise = xStart.getManager().randomNormal(xStart.getShape());
        }
        NDArray xNoisy = qSample(xStart, t, noise);
        NDArray xRecon = denoiseFn.apply(xNoisy, t);

        if ("noisepred".equals(lossType)) {
            // MSE between true noise and predicted noise, mean over spatial dims
            NDArray diff = noise.sub(xRecon);
            return NNUtils.meanflat(diff.square());
        }
        throw new UnsupportedOperationException("Unknown loss type: " + lossType);
    }

    // --- Reverse process (sampling) ---

    /** Compute p(x_{t-1} | x_t) mean and variance. */
    public NDArray[] pMeanVariance(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                                   NDArray x, NDArray t, boolean clipDenoised) {
        NDArray xRecon;
        if ("noisepred".equals(lossType)) {
            xRecon = predictStartFromNoise(x, t, denoiseFn.apply(x, t));
        } else {
            throw new UnsupportedOperationException(lossType);
        }
        if (clipDenoised) {
            xRecon = xRecon.clip(-1, 1);
        }
        NDArray[] post = qPosterior(xRecon, x, t);
        return post; // [mean, variance, logVariance]
    }

    /** Single reverse diffusion step. */
    public NDArray pSample(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                           NDArray x, NDArray t, boolean clipDenoised) {
        NDArray[] mv = pMeanVariance(denoiseFn, x, t, clipDenoised);
        NDArray modelMean = mv[0];
        NDArray modelLogVar = mv[2];

        NDArray noise = x.getManager().randomNormal(x.getShape());
        // No noise when t == 0
        NDArray nonzeroMask = t.eq(0).toType(DataType.FLOAT32, false).neg().add(1);
        long[] maskShape = new long[x.getShape().dimension()];
        java.util.Arrays.fill(maskShape, 1);
        maskShape[0] = x.getShape().get(0);
        nonzeroMask = nonzeroMask.reshape(maskShape);

        return modelMean.add(nonzeroMask.mul(modelLogVar.mul(0.5).exp()).mul(noise));
    }

    /**
     * Generate samples from pure noise via iterative denoising.
     * tf.while_loop replaced by Java for-loop (DJL eager execution).
     */
    public NDArray pSampleLoop(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                               NDManager manager, Shape shape) {
        NDArray img = manager.randomNormal(shape);
        for (int i = numTimesteps - 1; i >= 0; i--) {
            NDArray t = manager.full(new Shape(shape.get(0)), i).toType(DataType.INT32, false);
            img = pSample(denoiseFn, img, t, true);
        }
        return img;
    }

    /**
     * Generate samples and return the full denoising trajectory.
     *
     * @param repeatNoiseSteps number of initial steps using repeated noise across batch
     */
    public NDArray[] pSampleLoopTrajectory(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                                            NDManager manager, Shape shape,
                                            int repeatNoiseSteps) {
        boolean useRepeatNoise = (repeatNoiseSteps >= 0);
        NDArray noise;
        if (useRepeatNoise) {
            // Same noise for all batch elements
            long[] singleShape = shape.getShape().clone();
            singleShape[0] = 1;
            NDArray singleNoise = manager.randomNormal(new Shape(singleShape));
            noise = singleNoise.tile(new long[]{shape.get(0), 1, 1, 1});
        } else {
            noise = manager.randomNormal(shape);
        }

        java.util.List<NDArray> trajectory = new java.util.ArrayList<>();
        NDArray img = noise;
        trajectory.add(img);

        for (int i = numTimesteps - 1; i >= 0; i--) {
            NDArray t = manager.full(new Shape(shape.get(0)), i).toType(DataType.INT32, false);
            img = pSample(denoiseFn, img, t, true);
            trajectory.add(img);
        }

        return trajectory.toArray(new NDArray[0]);
    }

    // --- Interpolation ---

    /**
     * Interpolate between two images via diffusion.
     */
    public NDArray interpolate(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                               NDManager manager, NDArray x1, NDArray x2,
                               float lambda, int t) {
        NDArray tBatch = manager.full(new Shape(x1.getShape().get(0)), t)
                .toType(DataType.INT32, false);
        NDArray xt1 = qSample(x1, tBatch, null);
        NDArray xt2 = qSample(x2, tBatch, null);

        // Linear interpolation in latent space
        NDArray xtInterp = xt1.mul(1.0f - lambda).add(xt2.mul(lambda));

        // Reverse diffusion from t to 0
        for (int i = t; i >= 0; i--) {
            NDArray tStep = manager.full(new Shape(x1.getShape().get(0)), i)
                    .toType(DataType.INT32, false);
            xtInterp = pSample(denoiseFn, xtInterp, tStep, true);
        }
        return xtInterp;
    }

    // --- Double-precision helper methods ---

    private static double[] cumprod(double[] a) {
        double[] result = new double[a.length];
        result[0] = a[0];
        for (int i = 1; i < a.length; i++) result[i] = result[i - 1] * a[i];
        return result;
    }

    private static double[] sqrt(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = Math.sqrt(a[i]);
        return r;
    }

    private static double[] log(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = Math.log(a[i]);
        return r;
    }

    private static double[] oneMinus(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = 1.0 - a[i];
        return r;
    }

    private static double[] recip(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = 1.0 / a[i];
        return r;
    }

    private static double[] recipM1(double[] a) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = 1.0 / a[i] - 1.0;
        return r;
    }

    private static float[] toFloat(double[] a) {
        float[] r = new float[a.length];
        for (int i = 0; i < a.length; i++) r[i] = (float) a[i];
        return r;
    }
}
