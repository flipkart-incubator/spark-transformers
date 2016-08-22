package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ChiSqSelectorModelInfo;
import org.apache.spark.ml.feature.ChiSqSelectorModel;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link ChiSqSelectorModel} in MlLib to  {@link ChiSqSelectorModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class ChiSqSelectorModelInfoAdapter extends AbstractModelInfoAdapter<ChiSqSelectorModel, ChiSqSelectorModelInfo> {

    @Override
    public ChiSqSelectorModelInfo getModelInfo(final ChiSqSelectorModel from) {
        ChiSqSelectorModelInfo modelInfo = new ChiSqSelectorModelInfo();
        modelInfo.setSelectedFeatures(from.selectedFeatures());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getFeaturesCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<ChiSqSelectorModel> getSource() {
        return ChiSqSelectorModel.class;
    }

    @Override
    public Class<ChiSqSelectorModelInfo> getTarget() {
        return ChiSqSelectorModelInfo.class;
    }
}
