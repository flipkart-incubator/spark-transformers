package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.CustomLogScalerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a LogScaler model
 */
@Data
public class CustomLogScalerModelInfo extends AbstractModelInfo {

    private double addValue;

    /**
     * @return an corresponding {@link CustomLogScalerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new CustomLogScalerTransformer(this);
    }
}
