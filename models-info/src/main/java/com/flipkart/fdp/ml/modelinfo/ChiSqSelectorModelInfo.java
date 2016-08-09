package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.ChiSqSelectorTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a ChiSqSelector model
 */
@Data
public class ChiSqSelectorModelInfo extends AbstractModelInfo {
    private int[] selectedFeatures;

    /**
     * @return an corresponding {@link ChiSqSelectorTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new ChiSqSelectorTransformer(this);
    }
}
