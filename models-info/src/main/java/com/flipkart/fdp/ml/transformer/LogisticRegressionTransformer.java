package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Transforms input/ predicts for a Logistic Regression modelInfo representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo}.
 */
public class LogisticRegressionTransformer implements Transformer {
    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionTransformer.class);
    private final LogisticRegressionModelInfo modelInfo;

    public LogisticRegressionTransformer(final LogisticRegressionModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double predict(final double[] input) {
        double dotProduct = 0.0;
        for (int i = 0; i < input.length; i++) {
            dotProduct += input[i] * modelInfo.getWeights()[i];
        }
        double margin = dotProduct + modelInfo.getIntercept();
        double predictedRaw = 1.0 / (1.0 + Math.exp(-margin));
        return (predictedRaw > modelInfo.getThreshold() ? 1.0 : 0.0);
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(modelInfo.getInputKeys().iterator().next());
        input.put(modelInfo.getOutputKey(), predict(inp));
    }

}
