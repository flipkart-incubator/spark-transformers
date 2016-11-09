package com.flipkart.fdp.ml.modelinfo;

/**
 * Created by karan.verma on 09/11/16.
 */

import com.flipkart.fdp.ml.transformer.Transformer;
import com.flipkart.fdp.ml.transformer.VectorBinarizerTranformer;
import lombok.Data;

/**
 * Represents information for a Vector Binarizer model
 */
@Data
public class VectorBinarizerModelInfo extends AbstractModelInfo {

    private double threshold;

    /**
     * @return an corresponding {@link com.flipkart.fdp.ml.transformer.IfZeroVectorTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new VectorBinarizerTranformer(this);
    }

}
