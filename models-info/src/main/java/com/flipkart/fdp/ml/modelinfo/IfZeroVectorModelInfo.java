package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.IfZeroVectorTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a LogScaler model
 */
@Data
public class IfZeroVectorModelInfo extends AbstractModelInfo {

    private String thenSetValue;

    private String elseSetCol;

    /**
     * @return an corresponding {@link IfZeroVectorTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new IfZeroVectorTransformer(this);
    }
}
