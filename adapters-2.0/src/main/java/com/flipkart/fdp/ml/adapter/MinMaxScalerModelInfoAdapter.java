package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.MinMaxScalerModelInfo;
import org.apache.spark.ml.feature.MinMaxScalerModel;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link MinMaxScalerModel} in MlLib to  {@link MinMaxScalerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class MinMaxScalerModelInfoAdapter extends AbstractModelInfoAdapter<MinMaxScalerModel, MinMaxScalerModelInfo> {
    @Override
    public MinMaxScalerModelInfo getModelInfo(final MinMaxScalerModel from) {
        final MinMaxScalerModelInfo modelInfo = new MinMaxScalerModelInfo();
        modelInfo.setOriginalMax(from.originalMax().toArray());
        modelInfo.setOriginalMin(from.originalMin().toArray());
        modelInfo.setMax(from.getMax());
        modelInfo.setMin(from.getMin());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<MinMaxScalerModel> getSource() {
        return MinMaxScalerModel.class;
    }

    @Override
    public Class<MinMaxScalerModelInfo> getTarget() {
        return MinMaxScalerModelInfo.class;
    }
}
