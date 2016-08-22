package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link RandomForestModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class RandomForestModelInfoAdapter
        extends AbstractModelInfoAdapter<RandomForestModel, RandomForestModelInfo> {

    private final DecisionTreeModelInfoAdapter bridge = new DecisionTreeModelInfoAdapter();

    private RandomForestModelInfo visitForest(final RandomForestModel randomForestModel, DataFrame df) {
        final RandomForestModelInfo randomForestModelInfo = new RandomForestModelInfo();

        randomForestModelInfo.setAlgorithm(randomForestModel.algo().toString());

        final DecisionTreeModel[] decisionTreeModels = randomForestModel.trees();
        for (DecisionTreeModel i : decisionTreeModels) {
            DecisionTreeModelInfo tree = bridge.getModelInfo(i, df);
            randomForestModelInfo.getTrees().add(tree);
        }
        return randomForestModelInfo;
    }

    @Override
    public RandomForestModelInfo getModelInfo(RandomForestModel from, DataFrame df) {
        return visitForest(from, df);
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
