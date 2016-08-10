package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;


public abstract class AbstractModelInfoAdapter<F, T extends ModelInfo> implements ModelInfoAdapter<F, T> {

    @Override
    public T adapt(F from) {
        return getModelInfo(from);
    }

    /**
     * @param from source object in spark's mllib
     * @return returns the corresponding {@link ModelInfo} object that represents the model information
     */
    abstract T getModelInfo(F from);

}
