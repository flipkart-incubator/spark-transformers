package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo;

import java.util.Map;

/**
 * Transforms input/ predicts for a String Indexer model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo}.
 */
public class StringIndexerTransformer extends TransformerBase {

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
    public void transform(Map<String, Object> input) {
        String inp = (String) input.get(getInputKeys().iterator().next());
        input.put(getOutputKey(), predict(inp));
    }
}
