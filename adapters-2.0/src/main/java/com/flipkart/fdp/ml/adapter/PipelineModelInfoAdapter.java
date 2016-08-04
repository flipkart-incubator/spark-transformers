package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.ModelInfoAdapterFactory;
import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;

/**
 * Transforms Spark's {@link PipelineModel} to  {@link PipelineModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
public class PipelineModelInfoAdapter extends AbstractModelInfoAdapter<PipelineModel, PipelineModelInfo> {
    @Override
    public PipelineModelInfo getModelInfo(final PipelineModel from) {
        final PipelineModelInfo modelInfo = new PipelineModelInfo();
        final ModelInfo stages[] = new ModelInfo[from.stages().length];
        for (int i = 0; i < from.stages().length; i++) {
            Transformer sparkModel = from.stages()[i];
            stages[i] = ModelInfoAdapterFactory.getAdapter(sparkModel.getClass()).adapt(sparkModel);
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
