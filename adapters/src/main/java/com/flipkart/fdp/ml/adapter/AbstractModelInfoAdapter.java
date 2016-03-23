package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.utils.Constants;
import org.apache.spark.sql.DataFrame;


public abstract class AbstractModelInfoAdapter<F, T extends ModelInfo> implements ModelInfoAdapter<F, T> {

    private void preConditions(DataFrame df) {
        if(null != df) {
            if (!Constants.SUPPORTED_SPARK_VERSION.equals(df.sqlContext().sparkContext().version())) {
                throw new UnsupportedOperationException("Only spark version " + Constants.SUPPORTED_SPARK_VERSION + " is supported by this version of the library");
            }
        }
    }

    @Override
    public T adapt(F from, DataFrame df) {
        preConditions(df);
        return  getModelInfo(from, df);
    }

    /**
     * @param from source object in spark's mllib
     * @param df   Data frame that is used for training is required for some models as state information is being stored as column metadata by spark models
     * @return returns the corresponding {@link ModelInfo} object that represents the model information
     */
    abstract T getModelInfo(F from, DataFrame df);

}
