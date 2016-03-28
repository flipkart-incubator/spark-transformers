package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.Transformer;
import com.flipkart.fdp.ml.transformer.VectorAssemblerTransformer;

/**
 * Created by rohan.shetty on 28/03/16.
 */
public class VectorAssemblerModelInfo extends AbstractModelInfo {
    @Override
    public Transformer getTransformer() {
        return new VectorAssemblerTransformer(this);
    }
}
