package com.flipkart.fdp.ml.adapter;

/**
 * Transforms Spark's models in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo} object
 * that can be exported
 */

public interface ModelInfoAdapter<F, T> {

    T transform(F from);

    Class<F> getSource();

    Class<T> getTarget();
}
