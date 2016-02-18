package com.flipkart.fdp.ml.predictors;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RandomForestFPredictor implements Predictor<RandomForestModelInfo> {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForestFPredictor.class);
    private final RandomForestModelInfo forest;
    private final List<Predictor<DecisionTreeModelInfo>> subPredictors;

    public RandomForestFPredictor(RandomForestModelInfo forest) {
        this.forest = forest;
        this.subPredictors = new ArrayList<>(forest.trees.size());
        for (DecisionTreeModelInfo tree : forest.trees) {
            subPredictors.add(new DecisionTreePredictor(tree));
        }
    }

    public double predict(double[] input) {
        return predictForest(input);
    }

    private double predictForest(double[] input) {
        if (forest.algorithm.equals("Classification")) {
            return classify(input);
        } else {
            return regression(input);
        }
    }

    private double regression(double[] input) {
        double total = 0;
        for (Predictor<DecisionTreeModelInfo> i : subPredictors) {
            total += i.predict(input);
        }
        return total / subPredictors.size();
    }

    private double classify(double[] input) {
        Map<Double, Integer> votes = new HashMap<Double, Integer>();
        for (Predictor<DecisionTreeModelInfo> i : subPredictors) {
            double label = i.predict(input);

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
