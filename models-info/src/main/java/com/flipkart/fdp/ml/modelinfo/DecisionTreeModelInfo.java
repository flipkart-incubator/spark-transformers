package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.DecisionTreeTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Represents information for a Decision Tree model
 */
@Data
public class DecisionTreeModelInfo implements ModelInfo {
    private int root;
    private HashMap<Integer, Integer> leftChildMap = new HashMap<Integer, Integer>();
    private HashMap<Integer, Integer> rightChildMap = new HashMap<Integer, Integer>();
    private HashMap<Integer, DecisionNode> nodeInfo = new HashMap<Integer, DecisionNode>();

    /**
     * @return an corresponding {@link DecisionTreeTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new DecisionTreeTransformer(this);
    }

    @Data
    public static class DecisionNode {
        private int id;
        private int feature;
        private boolean isLeaf;
        private String featureType;
        private double threshold;
        private double predict;
        private double probability;
        private ArrayList<Double> categories;
    }

}
