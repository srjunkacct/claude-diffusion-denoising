package com.technodrome.diffusion.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;

/**
 * GPU memory management utilities.
 * Provides access to PyTorch's CUDA caching allocator via DJL's JNI layer.
 */
public class GpuMemoryUtils {

    private static final Logger logger = LoggerFactory.getLogger(GpuMemoryUtils.class);
    private static final Method EMPTY_CACHE_METHOD;

    static {
        Method m = null;
        try {
            Class<?> jniUtils = Class.forName("ai.djl.pytorch.jni.JniUtils");
            m = jniUtils.getDeclaredMethod("cudaEmptyCache");
            m.setAccessible(true);
        } catch (Exception e) {
            logger.debug("cudaEmptyCache not available: {}", e.getMessage());
        }
        EMPTY_CACHE_METHOD = m;
    }

    /**
     * Calls torch.cuda.empty_cache() via DJL's PyTorch JNI layer.
     * Releases unused cached memory from PyTorch's CUDA caching allocator.
     *
     * Should NOT be called every training step (hurts perf due to CUDA malloc overhead).
     * Use after sample generation or when switching between training and inference.
     */
    public static void cudaEmptyCache() {
        if (EMPTY_CACHE_METHOD != null) {
            try {
                EMPTY_CACHE_METHOD.invoke(null);
            } catch (Exception e) {
                logger.warn("cudaEmptyCache failed: {}", e.getMessage());
            }
        }
    }
}
