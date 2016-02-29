package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;

/**
 * Transforms a model ( eg spark's models in MlLib ) to  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo} object
 * that can be exported via {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public interface ModelInfoAdapter<F, T extends ModelInfo> {

    /**
     * @param from source object in spark's mllib
     * @return returns the corresponding {@link ModelInfo} object that represents the model information
     */
    T getModelInfo(F from);

    /**
     * @return Get the source class which is being adapted from.
     */
    Class<F> getSource();

    /**
     * @return Get the target adaptor class which is being adapted to.
     */
    Class<T> getTarget();
}
