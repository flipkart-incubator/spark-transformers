package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.BucketizerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a Bucketizer model
 */
@Data
public class BucketizerModelInfo implements ModelInfo{

    private double[] splits;

    /**
     * @return an corresponding {@link BucketizerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new BucketizerTransformer(this);
    }
}
