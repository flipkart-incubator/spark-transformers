package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.apache.commons.lang3.ArrayUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Transforms input/ predicts for a Random Forest model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo}.
 */
public class RandomForestTransformer implements Transformer {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForestTransformer.class);
    private static final String ALGO_CLASSIFICATION = "Classification";
    private static final String ALGO_REGRESSION = "Regression";
    private final RandomForestModelInfo forest;
    private final List<Transformer> subTransformers;

    public RandomForestTransformer(final RandomForestModelInfo forest) {
        this.forest = forest;
        this.subTransformers = new ArrayList<>(forest.getTrees().size());
        for (DecisionTreeModelInfo tree : forest.getTrees()) {
            subTransformers.add(new DecisionTreeTransformer(tree));
        }
    }

    public double predict(final double[] input) {
        return predictForest(input);
    }

    @Override
    public Object transform(Object[] input) {
        return predict(ArrayUtils.toPrimitive((Double [])input));
    }


    private double predictForest(final double[] input) {
        if (ALGO_CLASSIFICATION.equals(forest.getAlgorithm())) {
            return classify(input);
        } else if (ALGO_REGRESSION.equals(forest.getAlgorithm())) {
            return regression(input);
        } else {
            throw new UnsupportedOperationException("operation not supported for algo:" + forest.getAlgorithm());
        }
    }

    private double regression(final double[] input) {
        double total = 0;
        for (Transformer i : subTransformers) {
            total += (double)i.transform(ArrayUtils.toObject(input));
        }
        return total / subTransformers.size();
    }

    private double classify(final double[] input) {
        Map<Double, Integer> votes = new HashMap<Double, Integer>();
        for (Transformer i : subTransformers) {
            double label = (double)i.transform(ArrayUtils.toObject(input));

            Integer existingCount = votes.get(label);
            if (existingCount == null) {
                existingCount = 0;
            }

            int newCount = existingCount + 1;
            votes.put(label, newCount);
        }

        int maxVotes = 0;
        double maxVotesCandidate = 0;
        for (Map.Entry<Double, Integer> entry : votes.entrySet()) {
            if (entry.getValue() >= maxVotes) {
                maxVotes = entry.getValue();
                maxVotesCandidate = entry.getKey();
            }
        }
        return maxVotesCandidate;
    }
}
