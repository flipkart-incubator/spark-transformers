package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.CustomLogScaler;
import com.flipkart.fdp.ml.modelinfo.CustomLogScalerModelInfo;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms {@link CustomLogScaler} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.CustomLogScalerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class CustomLogScalerModelInfoAdapter extends AbstractModelInfoAdapter<CustomLogScaler, CustomLogScalerModelInfo> {

    @Override
    public CustomLogScalerModelInfo getModelInfo(final CustomLogScaler from, DataFrame df) {
        CustomLogScalerModelInfo modelInfo = new CustomLogScalerModelInfo();
        modelInfo.setAddValue(from.addValue());
        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);
        modelInfo.setOutputKey(from.getOutputCol());
        return modelInfo;
    }

    @Override
    public Class<CustomLogScaler> getSource() {
        return CustomLogScaler.class;
    }

    @Override
    public Class<CustomLogScalerModelInfo> getTarget() {
        return CustomLogScalerModelInfo.class;
    }
}
