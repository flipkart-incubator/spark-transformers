package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.LogisticRegressionTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a Logistic Regression model
 */
@Data
public class LogisticRegressionModelInfo extends AbstractModelInfo {
    private double[] weights;
    private double intercept;
    private int numClasses;
    private int numFeatures;
    private double threshold;

    /**
     * @return an corresponding {@link LogisticRegressionTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new LogisticRegressionTransformer(this);
    }
}
