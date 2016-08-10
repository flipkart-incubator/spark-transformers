package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.Log1PScalerModelInfo;

import java.util.Map;

/**
 * Transforms input/ predicts for a LogScaler model representation
 * captured by  {@link Log1PScalerModelInfo}.
 */
public class Log1PScalerTransformer implements Transformer {
    private final Log1PScalerModelInfo modelInfo;

    public Log1PScalerTransformer(Log1PScalerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(modelInfo.getInputKeys().iterator().next());
        input.put(modelInfo.getOutputKey(), predict(inp));
    }

    private double[] predict(double[] inp) {
        double[] output = new double[inp.length];
        for (int i = 0; i < inp.length; i++) {
            output[i] = Math.log1p(inp[i]);
        }
        return output;
    }
}
