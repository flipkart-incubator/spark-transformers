package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.VectorBinarizerModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Created by karan.verma on 09/11/16.
 */


public class VectorBinarizerTranformer implements Transformer {
    private final VectorBinarizerModelInfo modelInfo;

    public VectorBinarizerTranformer(final VectorBinarizerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    @Override
    public void transform(Map<String, Object> input) {
        Object value = input.get(modelInfo.getInputKeys().iterator().next());
        double[] inp = (value == null)? null: (double[])value;
        input.put(modelInfo.getOutputKeys().iterator().next(), predict(inp, modelInfo.getThreshold()));
    }

    private double[] predict(double[] inp, Double threshold) {

        if(inp == null || inp.length == 0) {
            return null;
        }

        double[] output = new double[inp.length];

        for(int i = 0; i < inp.length; i++) {
            double currentValue = inp[i];
            if (currentValue > threshold) {
                output[i] = 1.0;
            } else {
                output[i] = 0.0;
            }
        }
        return output;
    }

    @Override
    public Set<String> getInputKeys() {
        return modelInfo.getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return modelInfo.getOutputKeys();
    }
}
