package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.RandomForestTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;

import java.util.ArrayList;
/**
 * Represents information for a Random Forest model
 * */

public class RandomForestModelInfo implements ModelInfo {
    public String algorithm;
    public ArrayList<DecisionTreeModelInfo> trees = new ArrayList<DecisionTreeModelInfo>();

    public RandomForestModelInfo() {
    }

    /**
     * @return an corresponding {@link RandomForestTransformer} for this model info
     * */
    @Override
    public Transformer getTransformer() {
        return new RandomForestTransformer(this);
    }
}
