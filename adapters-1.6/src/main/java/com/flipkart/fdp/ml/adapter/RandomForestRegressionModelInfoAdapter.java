package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.tree.DecisionTreeModel;
import org.apache.spark.sql.DataFrame;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Transforms Spark's {@link org.apache.spark.ml.classification.RandomForestClassificationModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.RandomForestModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
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
    RandomForestModelInfo getModelInfo(final RandomForestRegressionModel sparkRfModel, final DataFrame df) {
        final RandomForestModelInfo modelInfo = new RandomForestModelInfo();

        modelInfo.setNumFeatures(sparkRfModel.numFeatures());
        modelInfo.setRegression(true); //true for regression

        final List<Double> treeWeights = new ArrayList<Double>();
        for (double w : sparkRfModel.treeWeights()) {
            treeWeights.add(w);
        }
        modelInfo.setTreeWeights(treeWeights);

        final List<DecisionTreeModelInfo> decisionTrees = new ArrayList<>();
        for (DecisionTreeModel decisionTreeModel : sparkRfModel.trees()) {
            decisionTrees.add(DECISION_TREE_ADAPTER.getModelInfo((DecisionTreeRegressionModel) decisionTreeModel, df));
        }
        modelInfo.setTrees(decisionTrees);

        final Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(sparkRfModel.getFeaturesCol());
        inputKeys.add(sparkRfModel.getLabelCol());
        modelInfo.setInputKeys(inputKeys);

        final Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(sparkRfModel.getPredictionCol());
        modelInfo.setOutputKeys(outputKeys);
        return modelInfo;
    }
}
