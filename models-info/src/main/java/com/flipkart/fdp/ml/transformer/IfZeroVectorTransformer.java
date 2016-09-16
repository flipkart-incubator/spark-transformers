package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.IfZeroVectorModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a IfZeroVector model representation
 * captured by  {@link IfZeroVectorModelInfo}.
 */
public class IfZeroVectorTransformer implements Transformer {

    private final IfZeroVectorModelInfo modelInfo;

    public IfZeroVectorTransformer(IfZeroVectorModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    @Override
    public void transform(Map<String, Object> input) {
        Object value = input.get(modelInfo.getInputKeys().iterator().next());
        double[] inp = (value == null)? null: (double[])value;
        String elseSetColValue = (String)input.get(modelInfo.getElseSetCol());
        input.put(modelInfo.getOutputKeys().iterator().next(), predict(inp, modelInfo.getThenSetValue(), elseSetColValue));
    }

    private String predict(double[] inp, String thenSetValue, String elseSetColValue) {
        if(inp == null || inp.length == 0) {
            return thenSetValue;
        }
        boolean allZero = true;
        for(int i=0; i<inp.length; i++) {
            //we don't need a double comparison tolerance here, since we are only detecting if these are sparse columns
            if(inp[i] != 0.0) {
                allZero=false;
                break;
            }
        }
        return allZero?thenSetValue:elseSetColValue;
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
