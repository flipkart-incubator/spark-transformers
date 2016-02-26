package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
 * Transforms input/ predicts for a Random Forest model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo}.
 * */
public class RandomForestTransformer implements Transformer {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForestTransformer.class);
    private final RandomForestModelInfo forest;
    private final List<Transformer> subTransformers;

    public RandomForestTransformer(RandomForestModelInfo forest) {
        this.forest = forest;
        this.subTransformers = new ArrayList<>(forest.trees.size());
        for (DecisionTreeModelInfo tree : forest.trees) {
            subTransformers.add(new DecisionTreeTransformer(tree));
        }
    }

    public double transform(double[] input) {
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
        for (Transformer i : subTransformers) {
            total += i.transform(input);
        }
        return total / subTransformers.size();
    }

    private double classify(double[] input) {
        Map<Double, Integer> votes = new HashMap<Double, Integer>();
        for (Transformer i : subTransformers) {
            double label = i.transform(input);

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
