package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.RandomForestFTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;

import java.util.ArrayList;

public class RandomForestModelInfo implements ModelInfo {
    public String algorithm;
    public ArrayList<DecisionTreeModelInfo> trees = new ArrayList<DecisionTreeModelInfo>();

    public RandomForestModelInfo() {
    }

    @Override
    public Transformer getTransformer() {
        return new RandomForestFTransformer(this);
    }
}
