package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.PipelineModelTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;


@Data
public class PipelineModelInfo implements ModelInfo{

    private ModelInfo stages[];

    @Override
    public Transformer getTransformer() {
        return new PipelineModelTransformer(this);
    }
}
