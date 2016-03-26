package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.StandardScalerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a StandardScaler model
 */

@Data
public class StandardScalerModelInfo extends AbstractModelInfo {
    private double[] std, mean;
    private boolean withStd, withMean;

    /**
     * @return an corresponding {@link StandardScalerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new StandardScalerTransformer(this);
    }
}
