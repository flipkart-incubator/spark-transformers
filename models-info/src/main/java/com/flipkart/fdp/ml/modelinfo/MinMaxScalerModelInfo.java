package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.MinMaxScalerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a MinMaxScaler model
 */

@Data
public class MinMaxScalerModelInfo implements ModelInfo {
    private double[] originalMin, originalMax;
    private double min, max;

    /**
     * @return an corresponding {@link MinMaxScalerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new MinMaxScalerTransformer(this);
    }
}
