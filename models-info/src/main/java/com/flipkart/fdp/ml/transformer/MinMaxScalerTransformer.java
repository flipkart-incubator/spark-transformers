package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.MinMaxScalerModelInfo;

import java.util.Map;

/**
 * Transforms input/ predicts for a MinMaxScaler model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.MinMaxScalerModelInfo}.
 */

public class MinMaxScalerTransformer extends TransformerBase {
    private final MinMaxScalerModelInfo modelInfo;

    public MinMaxScalerTransformer(final MinMaxScalerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    double[] predict(final double[] input) {
        //validate size of vectors
        if(modelInfo.getOriginalMax().length != modelInfo.getOriginalMin().length || modelInfo.getOriginalMax().length != input.length) {
            throw new IllegalArgumentException("Size of max, min and input vector are different : "
                    + modelInfo.getOriginalMax().length + " , " + modelInfo.getOriginalMin().length + " , " + input.length);
        }

        final double[] originalRange = new double[modelInfo.getOriginalMax().length];
        for( int i=0 ; i < originalRange.length; i++) {
            originalRange[i] = modelInfo.getOriginalMax()[i] - modelInfo.getOriginalMin()[i];
        }

        final double scale = modelInfo.getMax() - modelInfo.getMin();
        for( int i = 0; i < input.length; i++) {
            if(originalRange[i] != 0.0) {
                input[i] = ( input[i] - modelInfo.getOriginalMin()[i] ) / originalRange[i];
            }else{
                input[i] = 0.5;
            }
            input[i] = input[i]*scale + modelInfo.getMin();
        }
        return input;
    }

    @Override
    public void transform(Map<String, Object> input) {
        double inp[] = (double[]) input.get(getInputKeys()[0]);
        input.put(getOutputKey(), predict(inp));
    }
}
