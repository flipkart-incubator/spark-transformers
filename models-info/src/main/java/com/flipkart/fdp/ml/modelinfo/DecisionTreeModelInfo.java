package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.DecisionTreeTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Represents information for a Decision Tree model
 */
@Data
public class DecisionTreeModelInfo extends AbstractModelInfo {
    private DecisionNode root;
    private boolean continuousSplit;
    private String probabilityKey = "probability";
    private String rawPredictionKey = "rawPrediction";
    /**
     * @return an corresponding {@link DecisionTreeTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new DecisionTreeTransformer(this);
    }

    @Data
    public static class DecisionNode {
        private int feature;
        private boolean leaf;
        private double threshold;
        private double prediction;
        private List<Double> impurityStats = new ArrayList<>();
        private Set<Double> leftCategories = new HashSet<>();

        DecisionNode leftNode;
        DecisionNode rightNode;
    }

}
