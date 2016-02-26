package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.LogisticRegressionTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.ToString;

/**
 * Represents information for a Logistic Regression model
 * */

@ToString
public class LogisticRegressionModelInfo implements ModelInfo {
    public double[] weights;
    public double intercept;
    public int numClasses;
    public int numFeatures;

    /**
     * @return an corresponding {@link LogisticRegressionTransformer} for this model info
     * */
    @Override
    public Transformer getTransformer() {
        return new LogisticRegressionTransformer(this);
    }
}
