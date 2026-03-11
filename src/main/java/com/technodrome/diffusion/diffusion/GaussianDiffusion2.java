package com.technodrome.diffusion.diffusion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.technodrome.diffusion.nn.NNUtils;
import com.technodrome.diffusion.util.MathUtils;

import java.util.function.BiFunction;

/**
 * Extended Gaussian diffusion process.
 * Ported from diffusion_utils_2.py: GaussianDiffusion2.
 *
 * Supports:
 *   - model_mean_type: "xprev", "xstart", "eps"
 *   - model_var_type: "learned", "fixedsmall", "fixedlarge"
 *   - loss_type: "kl", "mse"
 *
 * Used by CIFAR-10 training (eps + fixedlarge + mse).
 * Stores coefficients as Java double arrays; creates NDArrays on-the-fly via extract().
 */
public class GaussianDiffusion2 {

    private final String modelMeanType;  // xprev, xstart, eps
    private final String modelVarType;   // learned, fixedsmall, fixedlarge
    private final String lossType;       // kl, mse
    public final int numTimesteps;

    // All coefficients stored as double arrays (converted to NDArray in _extract)
    private final double[] betas;
    private final double[] alphasCumprod;
    private final double[] alphasCumprodPrev;
    private final double[] sqrtAlphasCumprod;
    private final double[] sqrtOneMinusAlphasCumprod;
    private final double[] logOneMinusAlphasCumprod;
    private final double[] sqrtRecipAlphasCumprod;
    private final double[] sqrtRecipm1AlphasCumprod;
    private final double[] posteriorVariance;
    private final double[] posteriorLogVarianceClipped;
    private final double[] posteriorMeanCoef1;
    private final double[] posteriorMeanCoef2;

    public GaussianDiffusion2(double[] betasArr, String modelMeanType,
                              String modelVarType, String lossType) {
        this.modelMeanType = modelMeanType;
        this.modelVarType = modelVarType;
        this.lossType = lossType;
        this.numTimesteps = betasArr.length;

        this.betas = betasArr.clone();
        double[] alphas = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) alphas[i] = 1.0 - betasArr[i];

        this.alphasCumprod = cumprod(alphas);
        this.alphasCumprodPrev = new double[numTimesteps];
        alphasCumprodPrev[0] = 1.0;
        System.arraycopy(alphasCumprod, 0, alphasCumprodPrev, 1, numTimesteps - 1);

        sqrtAlphasCumprod = sqrt(alphasCumprod);
        sqrtOneMinusAlphasCumprod = sqrt(oneMinus(alphasCumprod));
        logOneMinusAlphasCumprod = log(oneMinus(alphasCumprod));
        sqrtRecipAlphasCumprod = sqrt(recip(alphasCumprod));
        sqrtRecipm1AlphasCumprod = sqrt(recipM1(alphasCumprod));

        posteriorVariance = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            posteriorVariance[i] = betasArr[i] * (1.0 - alphasCumprodPrev[i]) / (1.0 - alphasCumprod[i]);
        }

        // Note: GaussianDiffusion2 clips differently than GaussianDiffusion
        // posterior_log_variance_clipped = log(append(posterior_variance[1], posterior_variance[1:]))
        posteriorLogVarianceClipped = new double[numTimesteps];
        posteriorLogVarianceClipped[0] = Math.log(posteriorVariance[1]);
        for (int i = 1; i < numTimesteps; i++) {
            posteriorLogVarianceClipped[i] = Math.log(posteriorVariance[i]);
        }

        posteriorMeanCoef1 = new double[numTimesteps];
        posteriorMeanCoef2 = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            posteriorMeanCoef1[i] = betasArr[i] * Math.sqrt(alphasCumprodPrev[i]) / (1.0 - alphasCumprod[i]);
            posteriorMeanCoef2[i] = (1.0 - alphasCumprodPrev[i]) * Math.sqrt(alphas[i]) / (1.0 - alphasCumprod[i]);
        }
    }

    // --- Extract coefficient at timestep and broadcast ---
    private NDArray extract(double[] a, NDArray t, Shape xShape) {
        // Gather coefficients at timestep indices: a[t[i]] for each batch element
        long[] tLong = t.toType(DataType.INT64, false).toLongArray();
        float[] gathered = new float[tLong.length];
        for (int i = 0; i < tLong.length; i++) {
            gathered[i] = (float) a[(int) tLong[i]];
        }
        NDArray out = t.getManager().create(gathered).toDevice(t.getDevice(), false);
        long[] shape = new long[xShape.dimension()];
        java.util.Arrays.fill(shape, 1);
        shape[0] = xShape.get(0);
        return out.reshape(shape);
    }

    // --- Forward process ---

    public NDArray qSample(NDArray xStart, NDArray t, NDArray noise) {
        if (noise == null) {
            noise = xStart.getManager().randomNormal(xStart.getShape())
                    .toDevice(xStart.getDevice(), false);
        }
        return extract(sqrtAlphasCumprod, t, xStart.getShape()).mul(xStart)
                .add(extract(sqrtOneMinusAlphasCumprod, t, xStart.getShape()).mul(noise));
    }

    public NDArray[] qPosteriorMeanVariance(NDArray xStart, NDArray xT, NDArray t) {
        NDArray mean = extract(posteriorMeanCoef1, t, xT.getShape()).mul(xStart)
                .add(extract(posteriorMeanCoef2, t, xT.getShape()).mul(xT));
        NDArray var = extract(posteriorVariance, t, xT.getShape());
        NDArray logVarClipped = extract(posteriorLogVarianceClipped, t, xT.getShape());
        return new NDArray[]{mean, var, logVarClipped};
    }

    // --- Predict x_start ---

    private NDArray predictXstartFromEps(NDArray xT, NDArray t, NDArray eps) {
        return extract(sqrtRecipAlphasCumprod, t, xT.getShape()).mul(xT)
                .sub(extract(sqrtRecipm1AlphasCumprod, t, xT.getShape()).mul(eps));
    }

    private NDArray predictXstartFromXprev(NDArray xT, NDArray t, NDArray xprev) {
        // (xprev - coef2*x_t) / coef1
        double[] recipCoef1 = recip(posteriorMeanCoef1);
        double[] coef2overCoef1 = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++) {
            coef2overCoef1[i] = posteriorMeanCoef2[i] / posteriorMeanCoef1[i];
        }
        return extract(recipCoef1, t, xT.getShape()).mul(xprev)
                .sub(extract(coef2overCoef1, t, xT.getShape()).mul(xT));
    }

    // --- Reverse process ---

    /**
     * Compute p(x_{t-1} | x_t) mean and variance.
     *
     * @return [modelMean, modelVariance, modelLogVariance, predXstart]
     */
    public NDArray[] pMeanVariance(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                                   NDArray x, NDArray t, boolean clipDenoised) {
        Shape xShape = x.getShape();
        long B = xShape.get(0), C = xShape.get(1), H = xShape.get(2), W = xShape.get(3);
        NDArray modelOutput = denoiseFn.apply(x, t);

        // Variance
        NDArray modelVariance;
        NDArray modelLogVariance;
        switch (modelVarType) {
            case "learned":
                // Model outputs 2C channels: first C for mean, second C for log-variance
                NDList split = modelOutput.split(2, 1);
                modelOutput = split.get(0);
                modelLogVariance = split.get(1);
                modelVariance = modelLogVariance.exp();
                break;
            case "fixedlarge":
                modelVariance = extract(betas, t, xShape).mul(x.onesLike());
                double[] fixedLargeLogVar = new double[numTimesteps];
                fixedLargeLogVar[0] = Math.log(posteriorVariance[1]);
                for (int i = 1; i < numTimesteps; i++) {
                    fixedLargeLogVar[i] = Math.log(betas[i]);
                }
                modelLogVariance = extract(fixedLargeLogVar, t, xShape).mul(x.onesLike());
                break;
            case "fixedsmall":
                modelVariance = extract(posteriorVariance, t, xShape).mul(x.onesLike());
                modelLogVariance = extract(posteriorLogVarianceClipped, t, xShape).mul(x.onesLike());
                break;
            default:
                throw new UnsupportedOperationException("Unknown model_var_type: " + modelVarType);
        }

        // Mean parameterization
        NDArray predXstart;
        NDArray modelMean;
        switch (modelMeanType) {
            case "xprev":
                predXstart = predictXstartFromXprev(x, t, modelOutput);
                if (clipDenoised) predXstart = predXstart.clip(-1, 1);
                modelMean = modelOutput;
                break;
            case "xstart":
                predXstart = modelOutput;
                if (clipDenoised) predXstart = predXstart.clip(-1, 1);
                modelMean = qPosteriorMeanVariance(predXstart, x, t)[0];
                break;
            case "eps":
                predXstart = predictXstartFromEps(x, t, modelOutput);
                if (clipDenoised) predXstart = predXstart.clip(-1, 1);
                modelMean = qPosteriorMeanVariance(predXstart, x, t)[0];
                break;
            default:
                throw new UnsupportedOperationException("Unknown model_mean_type: " + modelMeanType);
        }

        return new NDArray[]{modelMean, modelVariance, modelLogVariance, predXstart};
    }

    /** Single reverse step. Returns [sample, predXstart]. */
    public NDArray[] pSample(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                             NDArray x, NDArray t, boolean clipDenoised) {
        NDArray[] mv = pMeanVariance(denoiseFn, x, t, clipDenoised);
        NDArray modelMean = mv[0];
        NDArray modelLogVar = mv[2];
        NDArray predXstart = mv[3];

        NDArray noise = x.getManager().randomNormal(x.getShape())
                .toDevice(x.getDevice(), false);
        // No noise when t == 0
        NDArray nonzeroMask = t.eq(0).toType(DataType.FLOAT32, false).neg().add(1);
        long[] maskShape = new long[x.getShape().dimension()];
        java.util.Arrays.fill(maskShape, 1);
        maskShape[0] = x.getShape().get(0);
        nonzeroMask = nonzeroMask.reshape(maskShape);

        NDArray sample = modelMean.add(nonzeroMask.mul(modelLogVar.mul(0.5).exp()).mul(noise));
        return new NDArray[]{sample, predXstart};
    }

    /** Generate samples from pure noise. */
    public NDArray pSampleLoop(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                               NDManager manager, Shape shape) {
        NDArray img = manager.randomNormal(shape);
        for (int i = numTimesteps - 1; i >= 0; i--) {
            // Use a sub-manager per denoising step so intermediates from each
            // forward pass (~50-100 MB) are freed immediately instead of
            // accumulating across all 1000 steps
            try (NDManager stepMgr = manager.newSubManager()) {
                img.attach(stepMgr);
                NDArray t = stepMgr.full(new Shape(shape.get(0)), i)
                        .toType(DataType.INT32, false);
                NDArray[] result = pSample(denoiseFn, img, t, true);
                img = result[0];
                img.attach(manager); // keep result alive for next step
            }
        }
        return img;
    }

    /**
     * Generate samples with progressive x_start predictions.
     *
     * @return [finalImage, xstartPredictions: [B, numRecorded, C, H, W]]
     */
    public NDArray[] pSampleLoopProgressive(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                                            NDManager manager, Shape shape,
                                            int includeXstartPredFreq) {
        NDArray img = manager.randomNormal(shape);
        int numRecorded = numTimesteps / includeXstartPredFreq;
        // xstartPreds: [B, numRecorded, C, H, W]
        long[] predsShape = new long[]{shape.get(0), numRecorded, shape.get(1), shape.get(2), shape.get(3)};
        NDArray xstartPreds = manager.zeros(new Shape(predsShape));

        for (int i = numTimesteps - 1; i >= 0; i--) {
            NDArray t = manager.full(new Shape(shape.get(0)), i).toType(DataType.INT32, false);
            NDArray[] result = pSample(denoiseFn, img, t, true);
            img = result[0];
            NDArray predXstart = result[1];

            // Record x_start prediction at specified frequency
            int slot = i / includeXstartPredFreq;
            if (slot < numRecorded && i % includeXstartPredFreq == 0) {
                // Insert predXstart into the correct slot
                // xstartPreds[:, slot, :, :, :] = predXstart
                for (int b = 0; b < shape.get(0); b++) {
                    xstartPreds.set(new ai.djl.ndarray.index.NDIndex("{},{}", b, slot),
                            predXstart.get(b));
                }
            }
        }
        return new NDArray[]{img, xstartPreds};
    }

    // --- Training loss ---

    /**
     * Compute training losses.
     *
     * @return per-sample losses [B]
     */
    public NDArray trainingLosses(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                                  NDArray xStart, NDArray t, NDArray noise) {
        if (noise == null) {
            noise = xStart.getManager().randomNormal(xStart.getShape())
                    .toDevice(xStart.getDevice(), false);
        }
        NDArray xT = qSample(xStart, t, noise);

        if ("kl".equals(lossType)) {
            return vbTermsBpd(denoiseFn, xStart, xT, t, false);
        } else if ("mse".equals(lossType)) {
            if ("learned".equals(modelVarType)) {
                throw new UnsupportedOperationException("MSE loss with learned variance not supported");
            }
            NDArray target;
            switch (modelMeanType) {
                case "xprev":
                    target = qPosteriorMeanVariance(xStart, xT, t)[0];
                    break;
                case "xstart":
                    target = xStart;
                    break;
                case "eps":
                    target = noise;
                    break;
                default:
                    throw new UnsupportedOperationException(modelMeanType);
            }
            NDArray modelOutput = denoiseFn.apply(xT, t);
            return NNUtils.meanflat(target.sub(modelOutput).square());
        }
        throw new UnsupportedOperationException("Unknown loss type: " + lossType);
    }

    // --- Variational bound ---

    /** VB terms in bits per dimension. */
    private NDArray vbTermsBpd(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                               NDArray xStart, NDArray xT, NDArray t, boolean clipDenoised) {
        NDArray[] truePosterior = qPosteriorMeanVariance(xStart, xT, t);
        NDArray trueMean = truePosterior[0];
        NDArray trueLogVar = truePosterior[2];

        NDArray[] modelPosterior = pMeanVariance(denoiseFn, xT, t, clipDenoised);
        NDArray modelMean = modelPosterior[0];
        NDArray modelLogVar = modelPosterior[2];

        NDArray kl = MathUtils.normalKl(trueMean, trueLogVar, modelMean, modelLogVar);
        kl = NNUtils.meanflat(kl).div(Math.log(2.0));

        NDArray decoderNll = MathUtils.discretizedGaussianLogLikelihood(
                xStart, modelMean, modelLogVar.mul(0.5)).neg();
        decoderNll = NNUtils.meanflat(decoderNll).div(Math.log(2.0));

        // At t=0, use decoder NLL; otherwise use KL
        return NDArrays.where(t.eq(0), decoderNll, kl);
    }

    /** Prior KL divergence. */
    public NDArray priorBpd(NDArray xStart) {
        long B = xStart.getShape().get(0);
        NDManager manager = xStart.getManager();
        NDArray t = manager.full(new Shape(B), numTimesteps - 1).toType(DataType.INT32, false);

        NDArray[] qMV = qMeanVariance(xStart, t);
        NDArray qtMean = qMV[0];
        NDArray qtLogVar = qMV[2];

        NDArray klPrior = MathUtils.normalKl(qtMean, qtLogVar,
                manager.zeros(qtMean.getShape()), manager.zeros(qtLogVar.getShape()));
        return NNUtils.meanflat(klPrior).div(Math.log(2.0));
    }

    private NDArray[] qMeanVariance(NDArray xStart, NDArray t) {
        NDArray mean = extract(sqrtAlphasCumprod, t, xStart.getShape()).mul(xStart);
        NDArray variance = extract(oneMinus(alphasCumprod), t, xStart.getShape());
        NDArray logVariance = extract(logOneMinusAlphasCumprod, t, xStart.getShape());
        return new NDArray[]{mean, variance, logVariance};
    }

    /**
     * Compute total BPD by looping over all timesteps.
     */
    public NDArray[] calcBpdLoop(BiFunction<NDArray, NDArray, NDArray> denoiseFn,
                                 NDArray xStart, boolean clipDenoised) {
        long B = xStart.getShape().get(0);
        NDManager manager = xStart.getManager();

        NDArray termsBpd = manager.zeros(new Shape(B, numTimesteps));
        NDArray mseBt = manager.zeros(new Shape(B, numTimesteps));

        for (int tVal = numTimesteps - 1; tVal >= 0; tVal--) {
            NDArray t = manager.full(new Shape(B), tVal).toType(DataType.INT32, false);
            NDArray xT = qSample(xStart, t, null);

            NDArray newVals = vbTermsBpd(denoiseFn, xStart, xT, t, clipDenoised);

            // MSE between predicted x_start and true x_start
            NDArray[] mv = pMeanVariance(denoiseFn, xT, t, clipDenoised);
            NDArray predXstart = mv[3];
            NDArray newMse = NNUtils.meanflat(predXstart.sub(xStart).square());

            // Insert into timestep slot
            // This is equivalent to the mask-based insertion in the Python code
            for (int b = 0; b < B; b++) {
                termsBpd.set(new ai.djl.ndarray.index.NDIndex("{},{}", b, tVal),
                        newVals.getFloat(b));
                mseBt.set(new ai.djl.ndarray.index.NDIndex("{},{}", b, tVal),
                        newMse.getFloat(b));
            }
        }

        NDArray priorBpdB = priorBpd(xStart);
        NDArray totalBpd = termsBpd.sum(new int[]{1}).add(priorBpdB);

        return new NDArray[]{totalBpd, termsBpd, priorBpdB, mseBt};
    }

    // --- Double-precision helpers ---

    private static double[] cumprod(double[] a) {
        double[] r = new double[a.length]; r[0] = a[0];
        for (int i = 1; i < a.length; i++) r[i] = r[i - 1] * a[i];
        return r;
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

}
