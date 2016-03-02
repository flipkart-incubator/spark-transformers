package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo;

/**
 * Created by akshay.us on 3/2/16.
 */
public class StringIndexerTransformer implements Transformer {

    private final StringIndexerModelInfo modelInfo;

    public StringIndexerTransformer(StringIndexerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double transform(String s) {
        Double index =  modelInfo.getLabelToIndex().get(s);
        if( null == index) {
            throw new RuntimeException("Unseen label :"+s);
        }
        return index;
    }

    @Override
    public double transform(double[] input) {
        throw new UnsupportedOperationException("transform on double values is not supported by StringIndexer");
    }
}
