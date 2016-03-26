package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms a model ( eg spark's models in MlLib ) to  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo} object
 * that can be exported via {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public interface ModelInfoAdapter<F, T extends ModelInfo> {

    /**
     * @param from source object in spark's mllib
     * @param df   Data frame that is used for training is required for some models as state information is being stored as column metadata by spark models
     * @return returns the corresponding {@link ModelInfo} object that represents the model information
     */
    T adapt(F from, DataFrame df);

    /**
     * @return Get the source class which is being adapted from.
     */
    Class<F> getSource();

    /**
     * @return Get the target adaptor class which is being adapted to.
     */
    Class<T> getTarget();
}
