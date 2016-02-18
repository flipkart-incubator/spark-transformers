package com.flipkart.fdp.ml.modelinfo.adapter;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

public class RandomForestModelInfoInfoAdapter
        implements ModelInfoAdapter<RandomForestModel, RandomForestModelInfo> {

    private DecisionTreeModelInfoInfoAdapter bridge = new DecisionTreeModelInfoInfoAdapter();

    private RandomForestModelInfo visitForest(RandomForestModel randomForestModel) {
        RandomForestModelInfo randomForestModelInfo = new RandomForestModelInfo();
        if (randomForestModel.algo().equals(Algo.Classification())) {
            randomForestModelInfo.algorithm = "Classification";
        }
        if (randomForestModel.algo().equals(Algo.Regression())) {
            randomForestModelInfo.algorithm = "Regression";
        }

        DecisionTreeModel[] decisionTreeModels = randomForestModel.trees();
        for (DecisionTreeModel i : decisionTreeModels) {
            DecisionTreeModelInfo tree = bridge.transform(i);
            randomForestModelInfo.trees.add(tree);
        }
        return randomForestModelInfo;
    }

    @Override
    public RandomForestModelInfo transform(RandomForestModel from) {
        RandomForestModelInfo randomForestModelInfoInfo = visitForest(from);
        return randomForestModelInfoInfo;
    }

    @Override
    public Class<RandomForestModel> getSource() {
        return RandomForestModel.class;
    }

    @Override
    public Class<RandomForestModelInfo> getTarget() {
        return RandomForestModelInfo.class;
    }
}
