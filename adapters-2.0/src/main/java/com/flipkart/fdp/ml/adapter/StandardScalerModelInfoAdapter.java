package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.StandardScalerModelInfo;
import org.apache.spark.ml.feature.StandardScalerModel;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link StandardScalerModel} in MlLib to  {@link StandardScalerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class StandardScalerModelInfoAdapter extends AbstractModelInfoAdapter<StandardScalerModel, StandardScalerModelInfo> {
    @Override
    public StandardScalerModelInfo getModelInfo(final StandardScalerModel from) {
        final StandardScalerModelInfo modelInfo = new StandardScalerModelInfo();
        modelInfo.setMean(from.mean().toArray());
        modelInfo.setStd(from.std().toArray());
        modelInfo.setWithMean(from.getWithMean());
        modelInfo.setWithStd(from.getWithStd());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<StandardScalerModel> getSource() {
        return StandardScalerModel.class;
    }

    @Override
    public Class<StandardScalerModelInfo> getTarget() {
        return StandardScalerModelInfo.class;
    }
}
