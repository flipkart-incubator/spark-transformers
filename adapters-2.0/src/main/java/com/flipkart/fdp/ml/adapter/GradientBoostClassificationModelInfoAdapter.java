package com.flipkart.fdp.ml.adapter;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.tree.DecisionTreeModel;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.GradientBoostModelInfo;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class GradientBoostClassificationModelInfoAdapter extends AbstractModelInfoAdapter<GBTClassificationModel, GradientBoostModelInfo> {

    private static final DecisionTreeRegressionModelInfoAdapter DECISION_TREE_ADAPTER = new DecisionTreeRegressionModelInfoAdapter();
    
    @Override
    GradientBoostModelInfo getModelInfo(final GBTClassificationModel sparkRfModel) {
        final GradientBoostModelInfo modelInfo = new GradientBoostModelInfo();

        modelInfo.setNumFeatures(sparkRfModel.numFeatures());
        modelInfo.setRegression(false); //false for classification

        final List<Double> treeWeights = new ArrayList<Double>();
        for (double w : sparkRfModel.treeWeights()) {
            treeWeights.add(w);
        }
        
        modelInfo.setTreeWeights(treeWeights);

        final List<DecisionTreeModelInfo> decisionTrees = new ArrayList<>();
        for (DecisionTreeModel decisionTreeModel : sparkRfModel.trees()) {
            decisionTrees.add(DECISION_TREE_ADAPTER.getModelInfo((DecisionTreeRegressionModel) decisionTreeModel));
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

    @Override
    public Class<GBTClassificationModel> getSource() {
        return GBTClassificationModel.class;
    }

    @Override
    public Class<GradientBoostModelInfo> getTarget() {
        return GradientBoostModelInfo.class;
    }

}
