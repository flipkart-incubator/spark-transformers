package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo;

import java.util.Arrays;
import java.util.Map;

/**
 * Transforms input/ predicts for a HashingTF model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo}.
 */
public class HashingTFTransformer extends TransformerBase {

    private final HashingTFModelInfo modelInfo;

    public HashingTFTransformer(final HashingTFModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double[] predict(final String[] terms) {
        final double[] encoding = new double[modelInfo.getNumFeatures()];
        Arrays.fill(encoding, 0.0);

        for (final String term : terms) {
            int index = term.hashCode() % modelInfo.getNumFeatures();
            //care for negative values
            if (index < 0) {
                index += modelInfo.getNumFeatures();
            }
            encoding[index] += 1.0;
        }
        return encoding;
    }

    @Override
    public void transform(Map<String, Object> input) {
        String [] inp = (String []) input.get(getInputKey());
        input.put(getOutputKey(), predict(inp));
    }

}
