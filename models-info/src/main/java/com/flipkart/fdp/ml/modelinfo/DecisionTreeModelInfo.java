package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.DecisionTreeTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.EqualsAndHashCode;

import java.util.ArrayList;
import java.util.HashMap;
/**
 * Represents information for a Decision Tree model
 * */

public class DecisionTreeModelInfo implements ModelInfo {
    public int root;
    public HashMap<Integer, Integer> leftChildMap = new HashMap<Integer, Integer>();
    public HashMap<Integer, Integer> rightChildMap = new HashMap<Integer, Integer>();
    public HashMap<Integer, DecisionNode> nodeInfo = new HashMap<Integer, DecisionNode>();

    public DecisionTreeModelInfo() {

    }

    /**
     * @return an corresponding {@link DecisionTreeTransformer} for this model info
     * */
    @Override
    public Transformer getTransformer() {
        return new DecisionTreeTransformer(this);
    }


    @EqualsAndHashCode
    public static class DecisionNode {
        public int id;
        public int feature;
        public boolean isLeaf;
        public String featureType;
        public double threshold;
        public double predict;
        public double probability;
        public ArrayList<Double> categories;

        public DecisionNode() {
        }
    }

}
