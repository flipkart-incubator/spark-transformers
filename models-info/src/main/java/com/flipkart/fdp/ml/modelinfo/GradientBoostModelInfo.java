package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.GradientBoostClassificationTransformer;
import com.flipkart.fdp.ml.transformer.RandomForestTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents information for a Random Forest model
 */

@Data
public class GradientBoostModelInfo extends AbstractModelInfo {

    private boolean regression;
    private int numFeatures;
    private List<DecisionTreeModelInfo> trees = new ArrayList<>();
    private List<Double> treeWeights = new ArrayList<>();

    private String probabilityKey = "probability";
    private String rawPredictionKey = "rawPrediction";

    /**
     * @return an corresponding {@link RandomForestTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new GradientBoostClassificationTransformer(this);
    }

    public boolean isClassification() {
        return !regression;
    }
}
