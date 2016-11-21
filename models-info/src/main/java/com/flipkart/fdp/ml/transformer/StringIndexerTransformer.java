package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a String Indexer model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo}.
 */
public class StringIndexerTransformer implements Transformer {

    private final StringIndexerModelInfo modelInfo;
    private final double maxIndex;

    public StringIndexerTransformer(final StringIndexerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
        //derive maximum index value to handleUnseen values
        double max = 0.0;
        for(Map.Entry<String, Double> entry : modelInfo.getLabelToIndex().entrySet()) {
            max = Math.max(max, entry.getValue());
        }
        maxIndex = max;
    }

    public double predict(final String s) {
        Double index = modelInfo.getLabelToIndex().get(s);
        if (null == index) {
            if(modelInfo.isFailOnUnseenValues()) {
                throw new RuntimeException("Unseen label :" + s);
            }else {
                //handling unseen value. Returning maxIndex+1
                index = maxIndex+1;
            }
        }
        return index;
    }

    @Override
    public void transform(Map<String, Object> input) {
        Object inp = input.get(modelInfo.getInputKeys().iterator().next());
        String stringInput = (null != inp)?inp.toString() : "";
        input.put(modelInfo.getOutputKeys().iterator().next(), predict(stringInput));
    }

    @Override
    public Set<String> getInputKeys() {
        return modelInfo.getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return modelInfo.getOutputKeys();
    }

}
