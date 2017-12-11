package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.utils.DecisionNodeAdapterUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.tree.Node;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;


/**
 * Transforms Spark's {@link org.apache.spark.ml.classification.DecisionTreeClassificationModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
public class DecisionTreeClassificationModelInfoAdapter
        extends AbstractModelInfoAdapter<DecisionTreeClassificationModel, DecisionTreeModelInfo> {

    public DecisionTreeModelInfo getModelInfo(final DecisionTreeClassificationModel decisionTreeModel,final DataFrame df) {
        final DecisionTreeModelInfo treeInfo = new DecisionTreeModelInfo();

        Node rootNode = decisionTreeModel.rootNode();
        treeInfo.setRoot(DecisionNodeAdapterUtils.adaptNode(rootNode));

        final Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(decisionTreeModel.getFeaturesCol());
        inputKeys.add(decisionTreeModel.getLabelCol());
        treeInfo.setInputKeys(inputKeys);

        final Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(decisionTreeModel.getPredictionCol());
        outputKeys.add(decisionTreeModel.getProbabilityCol());
        outputKeys.add(decisionTreeModel.getRawPredictionCol());
        treeInfo.setProbabilityKey(decisionTreeModel.getProbabilityCol());
        treeInfo.setRawPredictionKey(decisionTreeModel.getRawPredictionCol());
        treeInfo.setOutputKeys(outputKeys);

        return treeInfo;
    }

    @Override
    public Class<DecisionTreeClassificationModel> getSource() {
        return DecisionTreeClassificationModel.class;
    }

    @Override
    public Class<DecisionTreeModelInfo> getTarget() {
        return DecisionTreeModelInfo.class;
    }
}

