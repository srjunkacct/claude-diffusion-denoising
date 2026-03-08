package com.technodrome.diffusion.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.training.tracker.Tracker;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Manual Adam optimizer with explicit temporary NDArray cleanup.
 *
 * DJL's built-in Adam creates temporary NDArrays on the parent NDManager
 * during update() that are never closed, causing a slow Java heap leak.
 * This implementation closes every temporary via try-with-resources,
 * using in-place operations where possible to minimize allocations.
 *
 * Adam update rule:
 *   m = beta1 * m + (1 - beta1) * grad
 *   v = beta2 * v + (1 - beta2) * grad^2
 *   step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
 *   weight -= step_size * m / (sqrt(v) + eps)
 */
public class ManualAdam {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final Tracker lrTracker;

    // Per-parameter state: first and second moment estimates
    private final Map<String, NDArray> firstMoment = new LinkedHashMap<>();
    private final Map<String, NDArray> secondMoment = new LinkedHashMap<>();
    private int step = 0;

    public ManualAdam(Tracker lrTracker, float epsilon) {
        this(lrTracker, 0.9f, 0.999f, epsilon);
    }

    public ManualAdam(Tracker lrTracker, float beta1, float beta2, float epsilon) {
        this.lrTracker = lrTracker;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /**
     * Perform one Adam update step for all trainable parameters.
     * All intermediate NDArrays are explicitly closed to prevent heap leaks.
     *
     * IMPORTANT: We pre-create scalar constants and use NDArray overloads (e.g. muli(NDArray))
     * instead of primitive overloads (e.g. muli(float)). DJL's primitive overloads internally
     * call NDManager.create(n) to make a scalar that is NEVER closed, leaking on the parent
     * manager. Over thousands of steps this exhausts Java heap.
     */
    public void step(Block model) {
        step++;
        float lr = lrTracker.getNewValue(step);
        float stepSize = lr
                * (float) Math.sqrt(1.0 - Math.pow(beta2, step))
                / (1.0f - (float) Math.pow(beta1, step));

        // Pre-create scalar constants shared across all parameters
        NDManager mgr = getParamManager(model);
        if (mgr == null) return;

        try (NDArray beta1Scalar = mgr.create(beta1);
             NDArray oneMinusBeta1 = mgr.create(1.0f - beta1);
             NDArray beta2Scalar = mgr.create(beta2);
             NDArray oneMinusBeta2 = mgr.create(1.0f - beta2);
             NDArray epsScalar = mgr.create(epsilon);
             NDArray stepSizeScalar = mgr.create(stepSize)) {

            for (var pair : model.getParameters()) {
                var param = pair.getValue();
                if (!param.requiresGradient() || !param.getArray().hasGradient()) {
                    continue;
                }

                String id = pair.getKey();
                NDArray weight = param.getArray();

                // Initialize state on first step
                NDArray m = firstMoment.computeIfAbsent(id, k -> weight.zerosLike());
                NDArray v = secondMoment.computeIfAbsent(id, k -> weight.zerosLike());

                // Close gradient wrapper after use — getGradient() creates a NEW PtNDArray
                // on the parent manager each call (via manager.create(handle)), leaking if not closed.
                try (NDArray grad = weight.getGradient()) {
                    // m = beta1 * m + (1 - beta1) * grad
                    m.muli(beta1Scalar);
                    try (NDArray scaledGrad = grad.mul(oneMinusBeta1)) {
                        m.addi(scaledGrad);
                    }

                    // v = beta2 * v + (1 - beta2) * grad^2
                    v.muli(beta2Scalar);
                    try (NDArray gradSq = grad.square()) {
                        gradSq.muli(oneMinusBeta2);
                        v.addi(gradSq);
                    }
                }

                // weight -= step_size * m / (sqrt(v) + eps)
                try (NDArray sqrtV = v.sqrt()) {
                    sqrtV.addi(epsScalar);
                    try (NDArray update = m.div(sqrtV)) {
                        update.muli(stepSizeScalar);
                        weight.subi(update);
                    }
                }
            }
        }
    }

    private NDManager getParamManager(Block model) {
        for (var pair : model.getParameters()) {
            if (pair.getValue().requiresGradient()) {
                return pair.getValue().getArray().getManager();
            }
        }
        return null;
    }

    /** Close all state arrays (call when training is done). */
    public void close() {
        firstMoment.values().forEach(NDArray::close);
        secondMoment.values().forEach(NDArray::close);
        firstMoment.clear();
        secondMoment.clear();
    }

    public int getStep() {
        return step;
    }
}
