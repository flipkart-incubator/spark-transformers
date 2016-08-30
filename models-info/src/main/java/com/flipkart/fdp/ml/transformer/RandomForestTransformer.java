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
        for (final DecisionTreeModelInfo tree : forest.getTrees()) {
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
        if (forest.isClassification()) {
            //TODO: Optimize for double computation
            input.put(forest.getRawPredictionKey(), predictRaw(inp));
        }
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
        for (final DecisionTreeTransformer treeTransformer : subTransformers) {
            total += treeTransformer.predict(input);
        }
        return total / subTransformers.size();
    }

    private double classify(final double[] input) {
        return predictionFromProbabilities(normalizeToProbability(predictRaw(input)));
    }

    private double[] predictRaw(final double[] features) {
        final double[] votes = new double[forest.getNumClasses()];
        Arrays.fill(votes, 0.0);

        for (final DecisionTreeTransformer treeTransformer : subTransformers) {

            final double[] classCounts = treeTransformer.predictRaw(features);

            double total = 0.0;
            for (double val : classCounts) {
                total += val;
            }
            if (total != 0.0) {
                for (int i = 0; i < classCounts.length; i++) {
                    votes[i] += (classCounts[i] / total);
                }
            }
        }
        return votes;
    }

    private double[] normalizeToProbability(final double[] rawPrediction) {
        double total = 0.0;
        for (double val : rawPrediction) {
            total += val;
        }
        if (total != 0.0) {
            for (int i = 0; i < rawPrediction.length; i++) {
                rawPrediction[i] /= total;
            }
        }

        return rawPrediction;
    }

    private double predictionFromProbabilities(final double[] probabilities) {
        int max = 0;

        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[max]) {
                max = i;
            }
        }
        return (double) max;
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
