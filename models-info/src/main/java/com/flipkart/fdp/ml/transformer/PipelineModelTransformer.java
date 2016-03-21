package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Map;

/**
 * Transforms input/ predicts for a Pipeline model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.PipelineModelInfo}.
 */
public class PipelineModelTransformer extends TransformerBase {

    private final PipelineModelInfo modelInfo;
    private final Transformer transformers[];

    //TODO: support for non linear pipelines by deriving input and output column name from model being exported
    public PipelineModelTransformer(final PipelineModelInfo modelInfo) {
        this.modelInfo = modelInfo;
        transformers = new Transformer[modelInfo.getStages().length];
        for (int i = 0; i < transformers.length; i++) {
            transformers[i] = modelInfo.getStages()[i].getTransformer();
            transformers[i].setInputKeys(new LinkedHashSet<String>(Arrays.asList("output" + (i - 1))));
            transformers[i].setOutputKey("output" + i);
        }
        transformers[0].setInputKeys(new LinkedHashSet<String>(Arrays.asList("input")));
        transformers[transformers.length - 1].setOutputKey("output");
    }

    @Override
    public void transform(final Map<String, Object> input) {
        for (Transformer transformer : transformers) {
            transformer.transform(input);
        }
    }
}
