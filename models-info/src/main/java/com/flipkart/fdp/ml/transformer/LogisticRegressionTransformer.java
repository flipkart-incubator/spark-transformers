package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Transforms input/ predicts for a Logistic Regression model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo}.
 */
public class LogisticRegressionTransformer implements Transformer {
    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionTransformer.class);
    private final LogisticRegressionModelInfo model;

    public LogisticRegressionTransformer(LogisticRegressionModelInfo model) {
        this.model = model;
    }

    public double transform(double[] input) {
        double dotProduct = 0.0;
        for (int i = 0; i < input.length; i++) {
            dotProduct += input[i] * model.getWeights()[i];
        }
        double margin = dotProduct + model.getIntercept();
        return 1.0 / (1.0 + Math.exp(-margin));
    }

}
