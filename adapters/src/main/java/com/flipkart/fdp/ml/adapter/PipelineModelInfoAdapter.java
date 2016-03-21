package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.ModelInfoAdapterFactory;
import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.DataFrame;

/**
 * Created by akshay.us on 3/19/16.
 */
public class PipelineModelInfoAdapter implements ModelInfoAdapter<PipelineModel, PipelineModelInfo> {
    @Override
    public PipelineModelInfo getModelInfo(final PipelineModel from, final DataFrame df) {
        final PipelineModelInfo modelInfo = new PipelineModelInfo();
        ModelInfo stages [] = new ModelInfo[from.stages().length];
        for( int i = 0; i < from.stages().length; i++ ) {
            Transformer sparkModel = from.stages()[i];
            stages[i] = ModelInfoAdapterFactory.getAdapter(sparkModel.getClass()).getModelInfo(sparkModel,df);
        }
        modelInfo.setStages(stages);
        return modelInfo;
    }

    @Override
    public Class<PipelineModel> getSource() {
        return PipelineModel.class;
    }

    @Override
    public Class<PipelineModelInfo> getTarget() {
        return PipelineModelInfo.class;
    }
}
