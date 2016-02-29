package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.RandomForestTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.ArrayList;

/**
 * Represents information for a Random Forest model
 */

@Data
public class RandomForestModelInfo implements ModelInfo {
    private String algorithm;
    private ArrayList<DecisionTreeModelInfo> trees = new ArrayList<DecisionTreeModelInfo>();

    /**
     * @return an corresponding {@link RandomForestTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new RandomForestTransformer(this);
    }
}
