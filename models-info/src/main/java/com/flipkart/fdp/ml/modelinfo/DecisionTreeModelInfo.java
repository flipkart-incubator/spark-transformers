package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.DecisionTreeTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Represents information for a Decision Tree model. This class has been specifically designed to not contain and type heirarchy
 * for internal node/ leaf node , continuous/categorical split, regression/classification.
 * This has been done to keep serialization and deserialization of these objects simple.
 * Most of the json serializers (jackson, gson) do not handle type hierarchies well during deserialization.
 */
@Data
public class DecisionTreeModelInfo extends AbstractModelInfo {
    private DecisionNode root;
    //1.6 transformers refer this. Not removing for backward compatibility.
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
    public static class DecisionNode implements Serializable {
        private int feature;
        private boolean leaf;
        private double threshold;
        private double prediction;
        private List<Double> impurityStats = new ArrayList<>();
        private Set<Double> leftCategories = new HashSet<>();
        private boolean continuousSplit;

        DecisionNode leftNode;
        DecisionNode rightNode;
    }
}
