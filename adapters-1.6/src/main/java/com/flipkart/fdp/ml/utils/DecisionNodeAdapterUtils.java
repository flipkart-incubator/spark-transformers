package com.flipkart.fdp.ml.utils;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import org.apache.spark.ml.tree.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by akshay.us on 8/29/16.
 * Utility class for adapting a tree of {@link Node} to a tree of {@link com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo.DecisionNode}.
 */
public class DecisionNodeAdapterUtils  implements Serializable {

    public static DecisionTreeModelInfo.DecisionNode adaptNode(final Node node) {

        final DecisionTreeModelInfo.DecisionNode nodeInfo = new DecisionTreeModelInfo.DecisionNode();

        final List<Double> impurityStats = new ArrayList<>();
        for (double stat : node.impurityStats().stats()) {
            impurityStats.add(stat);
        }
        nodeInfo.setImpurityStats(impurityStats);
        nodeInfo.setPrediction(node.prediction());

        if (node instanceof LeafNode) {
            nodeInfo.setLeaf(true);
        } else {
            nodeInfo.setLeaf(false);
            final InternalNode internalNode = (InternalNode) node;
            nodeInfo.setFeature(internalNode.split().featureIndex());

            adaptIfCategoricalSplit(internalNode, nodeInfo);

            adaptIfContinuousSplit(internalNode, nodeInfo);

            if (internalNode.leftChild() != null) {
                final DecisionTreeModelInfo.DecisionNode leftNode = adaptNode(internalNode.leftChild());
                nodeInfo.setLeftNode(leftNode);
            }
            if (internalNode.rightChild() != null) {
                final DecisionTreeModelInfo.DecisionNode rightNode = adaptNode(internalNode.rightChild());
                nodeInfo.setRightNode(rightNode);
            }
        }
        return nodeInfo;
    }

    private static void adaptIfCategoricalSplit(final InternalNode internalNode, final DecisionTreeModelInfo.DecisionNode decisionNodeInfo) {
        if (internalNode.split() instanceof CategoricalSplit) {
            final CategoricalSplit categoricalSplit = (CategoricalSplit) internalNode.split();
            final Set<Double> leftCategories = new HashSet<>();
            //TODO: see if we need to optimise using right categories
            for (double c : categoricalSplit.leftCategories()) {
                leftCategories.add(c);
            }
            decisionNodeInfo.setLeftCategories(leftCategories);
        }
    }

    private static void adaptIfContinuousSplit(final InternalNode internalNode, final DecisionTreeModelInfo.DecisionNode decisionNodeInfo) {
        if (internalNode.split() instanceof ContinuousSplit) {
            final ContinuousSplit continuousSplit = (ContinuousSplit) internalNode.split();
            decisionNodeInfo.setThreshold(continuousSplit.threshold());
        }
    }

    public static boolean isContinuousSplit(final Node node) {
        if (!(node instanceof InternalNode)) {
            return false;
        }
        final InternalNode internalNode = (InternalNode) node;

        if (!(internalNode.split() instanceof ContinuousSplit)) {
            return false;
        }
        return true;
    }

}
