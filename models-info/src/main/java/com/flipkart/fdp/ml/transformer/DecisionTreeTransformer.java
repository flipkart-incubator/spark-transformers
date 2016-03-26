package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.AbstractModelInfo;
import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo.DecisionNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Transforms input/ predicts for a Decision Tree model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo}.
 */
public class DecisionTreeTransformer implements Transformer {
    private static final Logger LOG = LoggerFactory.getLogger(DecisionTreeTransformer.class);
    private static final String CONTINUOUS_FEATURE = "Continuous";
    private final DecisionTreeModelInfo tree;

    public DecisionTreeTransformer(final DecisionTreeModelInfo tree) {
        this.tree = tree;
    }

    private boolean visitLeft(final DecisionNode node, final double val) {
        return CONTINUOUS_FEATURE.equals(node.getFeatureType()) ?
                val <= node.getThreshold() :
                node.getCategories().contains(val);
    }

    private double predict(final DecisionNode node, final double[] input) {
        if (node.isLeaf()) {
            return node.getPredict();
        } else {
            final boolean visitLeft = visitLeft(node, input[node.getFeature()]);
            if (visitLeft) {
                DecisionNode leftChild = tree.getNodeInfo().get(tree.getLeftChildMap().get(node.getId()));
                return predict(leftChild, input);
            } else {
                DecisionNode rightChild = tree.getNodeInfo().get(tree.getRightChildMap().get(node.getId()));
                return predict(rightChild, input);
            }
        }
    }

    public double predict(final double[] input) {
        final DecisionNode node = tree.getNodeInfo().get(tree.getRoot());
        return predict(node, input);
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(tree.getInputKeys().iterator().next());
        input.put(tree.getOutputKey(), predict(inp));
    }

}
