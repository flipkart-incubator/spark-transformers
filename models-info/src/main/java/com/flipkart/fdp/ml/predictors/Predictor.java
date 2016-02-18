package com.flipkart.fdp.ml.predictors;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;

/**
 * This interface represents a capability of a class to predict the output of a model.
 * The model type and info is captured in the template parameter.
 */

public interface Predictor<T extends ModelInfo> {
    double predict(double[] input);
}
