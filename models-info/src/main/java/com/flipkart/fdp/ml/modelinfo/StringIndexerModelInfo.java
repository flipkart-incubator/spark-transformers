package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.StringIndexerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by akshay.us on 3/2/16.
 */
@Data
public class StringIndexerModelInfo implements ModelInfo{

    private Map<String, Double> labelToIndex = new HashMap<String, Double>();

    @Override
    public Transformer getTransformer() {
        return new StringIndexerTransformer(this);
    }
}
