package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

/**
 * Transforms Spark's {@link RandomForestModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class RandomForestModelInfoAdapter
        implements ModelInfoAdapter<RandomForestModel, RandomForestModelInfo> {

    private DecisionTreeModelInfoAdapter bridge = new DecisionTreeModelInfoAdapter();

    private RandomForestModelInfo visitForest(RandomForestModel randomForestModel) {
        RandomForestModelInfo randomForestModelInfo = new RandomForestModelInfo();

        randomForestModelInfo.setAlgorithm(randomForestModel.algo().toString());

        DecisionTreeModel[] decisionTreeModels = randomForestModel.trees();
        for (DecisionTreeModel i : decisionTreeModels) {
            DecisionTreeModelInfo tree = bridge.getModelInfo(i);
            randomForestModelInfo.getTrees().add(tree);
        }
        return randomForestModelInfo;
    }

    @Override
    public RandomForestModelInfo getModelInfo(RandomForestModel from) {
        return visitForest(from);
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
