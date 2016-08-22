package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a Pipeline model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.PipelineModelInfo}.
 */
public class PipelineModelTransformer implements Transformer {

    private final PipelineModelInfo modelInfo;
    private final Transformer transformers[];

    //TODO: support for non linear pipelines by deriving input and output column name from model being exported
    public PipelineModelTransformer(final PipelineModelInfo modelInfo) {
        this.modelInfo = modelInfo;
        transformers = new Transformer[modelInfo.getStages().length];
        for (int i = 0; i < transformers.length; i++) {
            transformers[i] = modelInfo.getStages()[i].getTransformer();
        }
    }

    @Override
    public void transform(final Map<String, Object> input) {
        for (Transformer transformer : transformers) {
            transformer.transform(input);
        }
    }

    @Override
    public Set<String> getInputKeys() {
        return transformers[0].getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return transformers[transformers.length - 1].getOutputKeys();
    }

}
