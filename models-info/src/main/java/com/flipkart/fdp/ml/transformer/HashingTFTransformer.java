package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo;

import java.util.Arrays;

/**
 * Created by akshay.us on 3/4/16.
 */
public class HashingTFTransformer implements Transformer {

    private HashingTFModelInfo modelInfo;

    public HashingTFTransformer(HashingTFModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double[] transform(String [] terms) {
        double [] encoding = new double[modelInfo.getNumFeatures()];
        Arrays.fill(encoding, 0.0);

        for(String term : terms) {
            int index = term.hashCode() % modelInfo.getNumFeatures();
            //care for negative values
            if(index < 0) {
                index += modelInfo.getNumFeatures();
            }
            encoding[index] += 1.0;
        }
        return encoding;
    }

    @Override
    public double transform(double[] input) {
        return 0;
    }
}
