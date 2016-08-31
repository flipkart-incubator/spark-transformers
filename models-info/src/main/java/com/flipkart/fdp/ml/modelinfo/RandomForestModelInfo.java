package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.RandomForestTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents information for a Random Forest model
 */

@Data
public class RandomForestModelInfo extends AbstractModelInfo {

    private boolean regression;
    private int numFeatures;
    private int numClasses;
    private List<DecisionTreeModelInfo> trees = new ArrayList<>();
    //Weights are currently not being used while prediction as it is not implemented in spark-mllib itself as of now. Keeping this as a placeholder for now.
    private List<Double> treeWeights = new ArrayList<>();

    private String probabilityKey = "probability";
    private String rawPredictionKey = "rawPrediction";

    /**
     * @return an corresponding {@link RandomForestTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new RandomForestTransformer(this);
    }

    public boolean isClassification() {
        return !regression;
    }
}
