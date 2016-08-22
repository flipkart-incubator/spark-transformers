package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.Log1PScalerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a LogScaler model
 */
@Data
public class Log1PScalerModelInfo extends AbstractModelInfo {

    /**
     * @return an corresponding {@link Log1PScalerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new Log1PScalerTransformer(this);
    }
}
