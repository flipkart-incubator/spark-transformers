package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ChiSqSelectorModelInfo;
import org.apache.spark.ml.feature.ChiSqSelectorModel;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link ChiSqSelectorModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.ChiSqSelectorModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class ChiSqSelectorModelInfoAdapter extends AbstractModelInfoAdapter<ChiSqSelectorModel, ChiSqSelectorModelInfo> {

    @Override
    public ChiSqSelectorModelInfo getModelInfo(final ChiSqSelectorModel from, DataFrame df) {
        ChiSqSelectorModelInfo modelInfo = new ChiSqSelectorModelInfo();
        modelInfo.setSelectedFeatures(from.selectedFeatures());
        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getFeaturesCol());
        modelInfo.setInputKeys(inputKeys);
        modelInfo.setOutputKey(from.getOutputCol());
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
