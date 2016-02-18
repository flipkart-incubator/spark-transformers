package com.flipkart.fdp.ml.predictors;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo.DecisionNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DecisionTreePredictor implements Predictor<DecisionTreeModelInfo> {
    private static final Logger LOG = LoggerFactory.getLogger(DecisionTreePredictor.class);

    private final DecisionTreeModelInfo tree;

    public DecisionTreePredictor(DecisionTreeModelInfo tree) {
        this.tree = tree;
    }

    private boolean visitLeft(DecisionNode node, double val) {
        return node.featureType.equals("Continuous") ?
                val <= node.threshold :
                node.categories.contains(val);
    }

    private double predict(DecisionNode node, double[] input) {
        if (node.isLeaf)
            return node.predict;
        else {
            boolean visitLeft = visitLeft(node, input[node.feature]);
            if (visitLeft) {
                DecisionNode leftChild = tree.nodeInfo.get(tree.leftChildMap.get(node.id));
                return predict(leftChild, input);
            } else {
                DecisionNode rightChild = tree.nodeInfo.get(tree.rightChildMap.get(node.id));
                return predict(rightChild, input);
            }
        }
    }

    public double predict(double[] input) {
        DecisionNode node = tree.nodeInfo.get(tree.root);
        return predict(node, input);
    }
}
