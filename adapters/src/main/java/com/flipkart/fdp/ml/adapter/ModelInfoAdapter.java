package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;

/**
 * Transforms Spark's models in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo} object
 * that can be exported
 */

public interface ModelInfoAdapter<F, T extends ModelInfo> {

    T getModelInfo(F from);

    Class<F> getSource();

    Class<T> getTarget();
}
