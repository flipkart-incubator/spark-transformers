package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo;

import java.util.Arrays;

/**
 * Transforms input/ predicts for a OneHotEncoder model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo}.
 */
public class OneHotEncoderTransformer implements Transformer {

    private final OneHotEncoderModelInfo modelInfo;

    public OneHotEncoderTransformer(final OneHotEncoderModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double[] predict(final double input) {
        final double encoding[];
        if (modelInfo.isShouldDropLast()) {
            encoding = new double[modelInfo.getNumTypes() - 1];
            Arrays.fill(encoding, 0.0);
            if ((int) input < modelInfo.getNumTypes()) {
                encoding[((int) input) - 1] = 1.0;
            }
        } else {
            encoding = new double[modelInfo.getNumTypes()];
            Arrays.fill(encoding, 0.0);
            encoding[((int) input) - 1] = 1.0;
        }
        return encoding;
    }

    @Override
    public Object transform(Object[] input) {
        if(input.length > 1) {
            throw new IllegalArgumentException("OneHotEncoderTransformer does not support arrays of length more than 1");
        }
        return predict((Double)input[0]);
    }
}
