package com.flipkart.fdp.ml.utils;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import org.apache.spark.ml.tree.*;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by akshay.us on 8/29/16.
 */
public class DecisionNodeAdapterUtils {

    public static DecisionTreeModelInfo.DecisionNode adaptNode(Node node) {

        final DecisionTreeModelInfo.DecisionNode nodeInfo = new DecisionTreeModelInfo.DecisionNode();

        if (node instanceof LeafNode) {
            nodeInfo.setLeaf(true);
            LeafNode leafNode = (LeafNode) node;
            nodeInfo.setPrediction(leafNode.prediction());
        } else {
            nodeInfo.setLeaf(false);
            InternalNode internalNode = (InternalNode) node;
            nodeInfo.setPrediction(internalNode.prediction());
            nodeInfo.setFeature(internalNode.split().featureIndex());

            adaptIfCategoricalSplit(internalNode, nodeInfo);

            adaptIfContinuousSplit(internalNode, nodeInfo);

            if (internalNode.leftChild() != null) {
                DecisionTreeModelInfo.DecisionNode leftNode = adaptNode(internalNode.leftChild());
                nodeInfo.setLeftNode(leftNode);
            }
            if (internalNode.rightChild() != null) {
                DecisionTreeModelInfo.DecisionNode rightNode = adaptNode(internalNode.rightChild());
                nodeInfo.setRightNode(rightNode);
            }
        }
        return nodeInfo;
    }

    private static void adaptIfCategoricalSplit(InternalNode internalNode, DecisionTreeModelInfo.DecisionNode decisionNodeInfo) {
        if (internalNode.split() instanceof CategoricalSplit) {
            CategoricalSplit categoricalSplit = (CategoricalSplit) internalNode.split();
            Set<Double> leftCategories = new HashSet<>();
            for (double c : categoricalSplit.leftCategories()) {
                leftCategories.add(c);
            }
            decisionNodeInfo.setLeftCategories(leftCategories);
        }
    }

    private static void adaptIfContinuousSplit(InternalNode internalNode, DecisionTreeModelInfo.DecisionNode decisionNodeInfo) {
        if (internalNode.split() instanceof ContinuousSplit) {
            ContinuousSplit continuousSplit = (ContinuousSplit) internalNode.split();
            decisionNodeInfo.setThreshold(continuousSplit.threshold());
        }
    }

    public static boolean isContinuousSplit(Node node) {
        if (!(node instanceof InternalNode)) {
            return false;
        }
        InternalNode internalNode = (InternalNode) node;

        if (!(internalNode.split() instanceof ContinuousSplit)) {
            return false;
        }
        return true;
    }

}
