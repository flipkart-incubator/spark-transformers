package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;
import com.flipkart.fdp.ml.utils.PipelineUtils;

import java.util.Map;
import java.util.Set;

/**
 * Transforms input/ predicts for a Pipeline model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.PipelineModelInfo}.
 */
public class PipelineModelTransformer implements Transformer {

    private final Transformer transformers[];
    private Set<String> extractedInputColumns;

    public PipelineModelTransformer(final PipelineModelInfo modelInfo) {
        transformers = new Transformer[modelInfo.getStages().length];
        for (int i = 0; i < transformers.length; i++) {
            transformers[i] = modelInfo.getStages()[i].getTransformer();
        }
        extractedInputColumns = PipelineUtils.extractRequiredInputColumns(transformers);
    }

    @Override
    public void transform(final Map<String, Object> input) {
        for (Transformer transformer : transformers) {
            transformer.transform(input);
        }
    }

    @Override
    public Set<String> getInputKeys() {
        return extractedInputColumns;
    }

    @Override
    public Set<String> getOutputKeys()
    {
        //using the output of last stage as output of pipeline.
        return transformers[transformers.length - 1].getOutputKeys();
    }

}
