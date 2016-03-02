package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.HashingTFTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Created by akshay.us on 3/4/16.
 */
@Data
public class HashingTFModelInfo implements ModelInfo {
    private int numFeatures;

    @Override
    public Transformer getTransformer() {
        return new HashingTFTransformer(this);
    }
}
