package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.CustomLogScalerModelInfo;

import java.util.Map;

/**
 * Transforms input/ predicts for a LogScaler model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.CustomLogScalerModelInfo}.
 */
public class CustomLogScalerTransformer implements Transformer {
    private final CustomLogScalerModelInfo modelInfo;

    public CustomLogScalerTransformer(CustomLogScalerModelInfo modelInfo) {
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
            output[i] = Math.log(inp[i] + modelInfo.getAddValue());
        }
        return output;
    }
}
