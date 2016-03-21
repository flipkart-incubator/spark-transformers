package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;

import java.util.Map;

/**
 * Created by akshay.us on 3/19/16.
 */
public class PipelineModelTransformer implements Transformer {

    private final PipelineModelInfo modelInfo;
    private final Transformer transformers[];

    public PipelineModelTransformer(final PipelineModelInfo modelInfo) {
        this.modelInfo = modelInfo;
        transformers = new Transformer[modelInfo.getStages().length];
        for( int i = 0;  i < transformers.length; i++) {
            transformers[i] = modelInfo.getStages()[i].getTransformer();
        }
    }

    @Override
    public void transform(Map<String, Object> input) {
            for(Transformer transformer : transformers) {
                transformer.transform(input);
            }
    }
}
