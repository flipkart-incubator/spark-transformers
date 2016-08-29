package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;

import java.util.*;

/**
 * Transforms input/ predicts for a Random Forest model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo}.
 */
public class RandomForestTransformer implements Transformer {
    private final RandomForestModelInfo forest;
    private final List<DecisionTreeTransformer> subTransformers;

    public RandomForestTransformer(final RandomForestModelInfo forest) {
        this.forest = forest;
        this.subTransformers = new ArrayList<>(forest.getTrees().size());
        for (DecisionTreeModelInfo tree : forest.getTrees()) {
            subTransformers.add((DecisionTreeTransformer) tree.getTransformer());
        }
    }

    public double predict(final double[] input) {
        return predictForest(input);
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(forest.getInputKeys().iterator().next());
        input.put(forest.getOutputKeys().iterator().next(), predict(inp));
    }


    private double predictForest(final double[] input) {
        if (forest.isClassification()) {
            return classify(input);
        } else {
            return regression(input);
        }
    }

    private double regression(final double[] input) {
        double total = 0.0;
        for (DecisionTreeTransformer i : subTransformers) {
            total += i.predict(input);
        }
        return total / subTransformers.size();
    }

    private double classify(final double[] input) {
        Map<Double, Integer> votes = new HashMap<Double, Integer>();
        for (DecisionTreeTransformer i : subTransformers) {
            double label = i.predict(input);

            if(votes.containsKey(label)) {
                votes.put(label, votes.get(label)+1);
            }else{
                votes.put(label, 0);
            }
        }

        int maxVotes = 0;
        double maxVotesCandidate = 0;
        for (Map.Entry<Double, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                maxVotesCandidate = entry.getKey();
            }
        }
        return maxVotesCandidate;
    }

    @Override
    public Set<String> getInputKeys() {
        return forest.getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return forest.getOutputKeys();
    }

}
