package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.tree.DecisionTreeModel;
import org.apache.spark.sql.DataFrame;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by akshay.us on 8/29/16.
 */
public class RandomForestRegressionModelInfoAdapter extends AbstractModelInfoAdapter<RandomForestRegressionModel, RandomForestModelInfo> {

    private static final DecisionTreeRegressionModelInfoAdapter DECISION_TREE_ADAPTER = new DecisionTreeRegressionModelInfoAdapter();

    @Override
    public Class<RandomForestRegressionModel> getSource() {
        return RandomForestRegressionModel.class;
    }

    @Override
    public Class<RandomForestModelInfo> getTarget() {
        return RandomForestModelInfo.class;
    }

    @Override
    RandomForestModelInfo getModelInfo(RandomForestRegressionModel sparkRfModel, DataFrame df) {
        RandomForestModelInfo modelInfo = new RandomForestModelInfo();

        modelInfo.setNumFeatures(sparkRfModel.numFeatures());
        modelInfo.setRegression(true); //true for regression

        List<Double> treeWeights = new ArrayList<Double>();
        for(double w : sparkRfModel.treeWeights()) {
            treeWeights.add(w);
        }
        modelInfo.setTreeWeights(treeWeights);

        List<DecisionTreeModelInfo> decisionTrees = new ArrayList<>();
        for(DecisionTreeModel decisionTreeModel: sparkRfModel.trees()) {
            decisionTrees.add( DECISION_TREE_ADAPTER.getModelInfo((DecisionTreeRegressionModel) decisionTreeModel, df));
        }
        modelInfo.setTrees(decisionTrees);

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(sparkRfModel.getFeaturesCol());
        inputKeys.add(sparkRfModel.getLabelCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(sparkRfModel.getPredictionCol());
        modelInfo.setOutputKeys(outputKeys);
        return modelInfo;
    }
}
