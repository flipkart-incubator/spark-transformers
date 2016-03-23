package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.StandardScalerModelInfo;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link StandardScalerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.StandardScalerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class StandardScalerModelInfoAdapter extends AbstractModelInfoAdapter<StandardScalerModel, StandardScalerModelInfo> {
    @Override
    public StandardScalerModelInfo getModelInfo(final StandardScalerModel from, final DataFrame df) {
        final StandardScalerModelInfo modelInfo = new StandardScalerModelInfo();
        modelInfo.setMean(from.mean().toArray());
        modelInfo.setStd(from.std().toArray());
        modelInfo.setWithMean(from.getWithMean());
        modelInfo.setWithStd(from.getWithStd());
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
