package com.technodrome.diffusion.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Exponential Moving Average of model parameters.
 * Maintains shadow copies of all trainable parameters and updates them with:
 *   shadow = decay * shadow + (1 - decay) * param
 *
 * Ported from utils.py: ema_scope() and tpu_utils.py: make_ema().
 */
public class EMAHelper {

    private final float decay;
    private final Map<String, NDArray> shadowParams = new LinkedHashMap<>();
    private final Block model;

    public EMAHelper(Block model, float decay) {
        this.model = model;
        this.decay = decay;
    }

    /** Initialize shadow parameters as copies of current model parameters. */
    public void register(NDManager manager) {
        for (var pair : model.getParameters()) {
            Parameter param = pair.getValue();
            if (param.requiresGradient()) {
                NDArray value = param.getArray();
                shadowParams.put(pair.getKey(), value.duplicate());
            }
        }
    }

    /** Update shadow parameters with EMA decay. Call after each optimizer step. */
    public void update() {
        if (shadowParams.isEmpty()) return;

        // Pre-create scalar constants to avoid per-param leak from muli(Number).
        // DJL's muli(Number) internally creates a scalar NDArray that is never closed.
        NDManager mgr = shadowParams.values().iterator().next().getManager();
        try (NDArray decayScalar = mgr.create(decay);
             NDArray oneMinusDecay = mgr.create(1.0f - decay)) {
            for (var pair : model.getParameters()) {
                Parameter param = pair.getValue();
                if (param.requiresGradient()) {
                    NDArray shadow = shadowParams.get(pair.getKey());
                    if (shadow != null) {
                        NDArray current = param.getArray();
                        // shadow = decay * shadow + (1 - decay) * current
                        shadow.muli(decayScalar);
                        try (NDArray scaled = current.mul(oneMinusDecay)) {
                            shadow.addi(scaled);
                        }
                    }
                }
            }
        }
    }

    /**
     * In-place swap of model parameters ↔ shadow (EMA) parameters.
     * Call once before sampling (param → EMA), call again after (EMA → param).
     *
     * Uses only GPU-native in-place ops — no duplicate(), toFloatArray(), or CPU transfer.
     * This avoids leaking PtNDManagers on the parent manager.
     */
    public void swapWithEma() {
        for (var pair : model.getParameters()) {
            Parameter param = pair.getValue();
            if (param.requiresGradient()) {
                NDArray shadow = shadowParams.get(pair.getKey());
                if (shadow != null) {
                    NDArray paramArray = param.getArray();
                    // In-place swap: param ↔ shadow
                    // diff = param - shadow (temp, closed after use)
                    // param -= diff → param = shadow ✓
                    // shadow += diff → shadow = param_orig ✓
                    try (NDArray diff = paramArray.sub(shadow)) {
                        paramArray.subi(diff);
                        shadow.addi(diff);
                    }
                }
            }
        }
    }

    public float getDecay() {
        return decay;
    }
}
