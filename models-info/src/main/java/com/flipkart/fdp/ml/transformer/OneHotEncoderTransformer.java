package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo;
import org.apache.commons.lang3.ArrayUtils;

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
    public Object[] transform(Object[] input) {
        if (input.length > 1) {
            throw new IllegalArgumentException("OneHotEncoderTransformer does not support arrays of length more than 1");
        }
        return ArrayUtils.toObject(predict((Double) input[0]));
    }
}
