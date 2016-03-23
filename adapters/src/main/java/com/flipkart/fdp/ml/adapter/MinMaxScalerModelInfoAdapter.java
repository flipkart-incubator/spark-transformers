package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.MinMaxScalerModelInfo;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link MinMaxScalerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.MinMaxScalerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class MinMaxScalerModelInfoAdapter extends AbstractModelInfoAdapter<MinMaxScalerModel, MinMaxScalerModelInfo> {
    @Override
    public MinMaxScalerModelInfo getModelInfo(final MinMaxScalerModel from, final DataFrame df) {
        final MinMaxScalerModelInfo modelInfo = new MinMaxScalerModelInfo();
        modelInfo.setOriginalMax(from.originalMax().toArray());
        modelInfo.setOriginalMin(from.originalMin().toArray());
        modelInfo.setMax(from.getMax());
        modelInfo.setMin(from.getMin());
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
