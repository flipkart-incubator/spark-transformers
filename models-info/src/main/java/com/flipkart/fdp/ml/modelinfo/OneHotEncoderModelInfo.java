package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.OneHotEncoderTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a one hot encoder model
 */
@Data
public class OneHotEncoderModelInfo extends AbstractModelInfo {

    private int numTypes;

    /**
     * @return an corresponding {@link OneHotEncoderTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new OneHotEncoderTransformer(this);
    }
}
