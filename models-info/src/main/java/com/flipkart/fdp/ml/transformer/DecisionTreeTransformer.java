package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo.DecisionNode;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a Decision Tree model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo}.
 */
public class DecisionTreeTransformer implements Transformer {

    private final DecisionTreeModelInfo tree;

    public DecisionTreeTransformer(final DecisionTreeModelInfo tree) {
        this.tree = tree;
    }

    public double predict(final double[] input) {
        final DecisionNode node = tree.getRoot();
        return predict(node, input);
    }

    private double predict(final DecisionNode node, final double[] input) {
        if (node.isLeaf()) {
            return node.getPrediction();
        } else {
            final boolean shouldGoLeft = shouldGoLeft(node, input[node.getFeature()]);
            if (shouldGoLeft) {
                DecisionNode leftChild = node.getLeftNode();
                return predict(leftChild, input);
            } else {
                DecisionNode rightChild = node.getRightNode();
                return predict(rightChild, input);
            }
        }
    }

    public double[] predictRaw(final double[] input) {
        final DecisionNode node = tree.getRoot();
        return predictRaw(node, input);
    }

    private double[] predictRaw(final DecisionNode node, final double[] input) {
        if (node.isLeaf()) {
            double[] rawPrediction = new double[node.getImpurityStats().size()];
            for (int i = 0; i < rawPrediction.length; i++) {
                rawPrediction[i] = node.getImpurityStats().get(i);
            }
            return rawPrediction;
        } else {
            final boolean shouldGoLeft = shouldGoLeft(node, input[node.getFeature()]);
            if (shouldGoLeft) {
                final DecisionNode leftChild = node.getLeftNode();
                return predictRaw(leftChild, input);
            } else {
                final DecisionNode rightChild = node.getRightNode();
                return predictRaw(rightChild, input);
            }
        }
    }

    private boolean shouldGoLeft(final DecisionNode node, final double val) {
        //Using the isContinuous property at node, rather thn the root. 
    	if (node.isContinuousSplit()) {
            return val <= node.getThreshold();
        } else {
            return node.getLeftCategories().contains(val);
        }
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(tree.getInputKeys().iterator().next());
        input.put(tree.getOutputKeys().iterator().next(), predict(inp));
        //TODO: optimise for double computation
        input.put(tree.getRawPredictionKey(), predictRaw(inp));
    }

    @Override
    public Set<String> getInputKeys() {
        return tree.getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return tree.getOutputKeys();
    }

}
