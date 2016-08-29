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

    public DecisionTreeModelInfo getModelInfo(final DecisionTreeClassificationModel decisionTreeModel, DataFrame df) {
        final DecisionTreeModelInfo treeInfo = new DecisionTreeModelInfo();

        //TODO: verify this is correct. Extracting split type for entire decision tree from root node
        treeInfo.setContinuousSplit( DecisionNodeAdapterUtils.isContinuousSplit(decisionTreeModel.rootNode()));

        Node rootNode = decisionTreeModel.rootNode();
        treeInfo.setRoot( DecisionNodeAdapterUtils.adaptNode(rootNode));

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(decisionTreeModel.getFeaturesCol());
        inputKeys.add(decisionTreeModel.getLabelCol());
        treeInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(decisionTreeModel.getPredictionCol());
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

