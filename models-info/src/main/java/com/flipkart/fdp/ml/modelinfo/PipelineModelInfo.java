package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.PipelineModelTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a pipeline model
 */
@Data
public class PipelineModelInfo implements ModelInfo {

    private ModelInfo stages[];

    /**
     * @return an corresponding {@link PipelineModelTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new PipelineModelTransformer(this);
    }
}
