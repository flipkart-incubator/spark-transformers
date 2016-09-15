package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.FillNAValuesTransformerModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a {@link FillNAValuesTransformerModelInfo} model representation
 * captured by  {@link FillNAValuesTransformerModelInfo}.
 */
public class FillNAValuesTransformer implements Transformer {
    private final FillNAValuesTransformerModelInfo modelInfo;

    public FillNAValuesTransformer(FillNAValuesTransformerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    @Override
    public void transform(Map<String, Object> input) {
        for(Map.Entry<String, Object> entry : modelInfo.getNaValuesMap().entrySet()) {
            if( isNA(input.get(entry.getKey()))) {
                input.put(entry.getKey(), entry.getValue());
            }
        }
    }

    private boolean isNA(Object data) {
        if( null == data) {
            return true;
        }
        if( data instanceof Double) {
            return ((Double)data).isNaN();
        }
        if( data instanceof Float) {
            return ((Float)data).isNaN();
        }
        return false;
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
