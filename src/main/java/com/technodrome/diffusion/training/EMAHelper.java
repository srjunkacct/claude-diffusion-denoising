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
        for (var pair : model.getParameters()) {
            Parameter param = pair.getValue();
            if (param.requiresGradient()) {
                NDArray shadow = shadowParams.get(pair.getKey());
                if (shadow != null) {
                    NDArray current = param.getArray();
                    // shadow = decay * shadow + (1 - decay) * current
                    shadow.muli(decay).addi(current.mul(1.0f - decay));
                }
            }
        }
    }

    /**
     * Swap shadow parameters into the model for inference.
     * Returns the original parameters so they can be restored.
     */
    public Map<String, NDArray> swapToEma() {
        Map<String, NDArray> originals = new LinkedHashMap<>();
        for (var pair : model.getParameters()) {
            Parameter param = pair.getValue();
            if (param.requiresGradient()) {
                NDArray shadow = shadowParams.get(pair.getKey());
                if (shadow != null) {
                    NDArray original = param.getArray().duplicate();
                    originals.put(pair.getKey(), original);
                    param.getArray().set(shadow.toFloatArray());
                }
            }
        }
        return originals;
    }

    /** Restore original parameters after EMA inference. */
    public void restoreFromEma(Map<String, NDArray> originals) {
        for (var pair : model.getParameters()) {
            Parameter param = pair.getValue();
            NDArray original = originals.get(pair.getKey());
            if (original != null) {
                param.getArray().set(original.toFloatArray());
                original.close();
            }
        }
    }

    public float getDecay() {
        return decay;
    }
}
