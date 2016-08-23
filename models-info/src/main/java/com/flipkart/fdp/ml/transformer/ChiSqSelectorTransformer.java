package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.ChiSqSelectorModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a ChiSqSelectorModel model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.ChiSqSelectorModelInfo}.
 */
public class ChiSqSelectorTransformer implements Transformer {

    private final ChiSqSelectorModelInfo modelInfo;

    public ChiSqSelectorTransformer(final ChiSqSelectorModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double[] predict(double[] inp) {
        double[] output = new double[modelInfo.getSelectedFeatures().length];
        int count = 0;
        for (int featureIndex : modelInfo.getSelectedFeatures()) {
            output[count++] = inp[featureIndex];
        }
        return output;
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(modelInfo.getInputKeys().iterator().next());
        input.put(modelInfo.getOutputKeys().iterator().next(), predict(inp));
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
