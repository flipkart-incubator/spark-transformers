package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.HashingTFTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a HashingTF model
 */
@Data
public class HashingTFModelInfo implements ModelInfo {
    private int numFeatures;

    /**
     * @return an corresponding {@link HashingTFTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new HashingTFTransformer(this);
    }
}
