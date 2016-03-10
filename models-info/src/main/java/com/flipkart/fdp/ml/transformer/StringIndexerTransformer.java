package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo;

/**
 * Transforms input/ predicts for a String Indexer model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo}.
 */
public class StringIndexerTransformer implements Transformer {

    private final StringIndexerModelInfo modelInfo;

    public StringIndexerTransformer(final StringIndexerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double predict(final String s) {
        final Double index = modelInfo.getLabelToIndex().get(s);
        if (null == index) {
            throw new RuntimeException("Unseen label :" + s);
        }
        return index;
    }

    @Override
    public Object transform(Object[] input) {
        if(input.length > 1) {
            throw new IllegalArgumentException("StringIndexerTransformer does not support arrays of length more than 1");
        }
        return predict((String)input[0]);
    }
}
