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

    public double[] transform(final double input) {
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

    //TODO: what finalise on the Transformer interface
    @Override
    public double transform(double[] input) {
        return 0;
    }
}
