package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo;

import java.util.Arrays;

/**
 * Created by akshay.us on 3/4/16.
 */
public class OneHotEncoderTransformer implements Transformer {

    private final OneHotEncoderModelInfo modelInfo;

    public OneHotEncoderTransformer(OneHotEncoderModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double[] transform(double input) {
        double encoding[];
        if(modelInfo.isShouldDropLast()) {
            encoding = new double[modelInfo.getNumTypes()-1];
            Arrays.fill(encoding, 0.0);
            if((int)input < modelInfo.getNumTypes()) {
                encoding[((int) input)-1] = 1.0;
            }
        }else{
            encoding = new double[modelInfo.getNumTypes()];
            Arrays.fill(encoding, 0.0);
            encoding[((int) input)-1] = 1.0;
        }
        return encoding;
    }

    @Override
    public double transform(double[] input) {
        return 0;
    }
}
