package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;

/**
 * Transforms Spark's models in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo} object
 * that can be exported via {@link com.flipkart.fdp.ml.SparkModelExporter}
 */
public interface ModelInfoAdapter<F, T extends ModelInfo> {

    /**
     *@return returns the corresponding {@link ModelInfo} object that represents the model information
     *@param from source object in spark's mllib
     * */
    T getModelInfo(F from);

    /**
     * @return Get the source class which is being adapted from.
     * */
    Class<F> getSource();

    /**
     * @return Get the target adaptor class which is being adapted to.
     * */
    Class<T> getTarget();
}
