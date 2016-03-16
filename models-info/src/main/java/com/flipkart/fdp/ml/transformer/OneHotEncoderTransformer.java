package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo;

import java.util.Arrays;
import java.util.Map;

/**
 * Transforms input/ predicts for a OneHotEncoder model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo}.
 */
public class OneHotEncoderTransformer extends TransformerBase {

    private final OneHotEncoderModelInfo modelInfo;

    public OneHotEncoderTransformer(final OneHotEncoderModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double[] predict(final double input) {
        int size = modelInfo.getNumTypes();
        if (modelInfo.isShouldDropLast()) {
            size--;
        }
        final double encoding[] = new double[size];
        Arrays.fill(encoding, 0.0);

        if ((int) input < size) {
            encoding[((int) input)] = 1.0;
        }
        return encoding;
    }

    @Override
    public void transform(Map<String, Object> input) {
        double inp = (double) input.get(getInputKey());
        input.put(getOutputKey(), predict(inp));
    }
}
