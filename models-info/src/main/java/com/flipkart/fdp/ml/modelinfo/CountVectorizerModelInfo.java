package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.CountVectorizerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a CountVectorizer model
 */
@Data
public class CountVectorizerModelInfo extends AbstractModelInfo {

    private double minTF;
    private String[] vocabulary;

    /**
     * @return an corresponding {@link CountVectorizerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new CountVectorizerTransformer(this);
    }
}
