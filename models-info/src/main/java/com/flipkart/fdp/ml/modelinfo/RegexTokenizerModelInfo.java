package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.RegexTokenizerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Represents information for a RegexTokenizer model
 */

@Data
public class RegexTokenizerModelInfo implements ModelInfo {
    private int minTokenLength;
    private boolean gaps, toLowercase;
    private String pattern;

    /**
     * @return an corresponding {@link RegexTokenizerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new RegexTokenizerTransformer(this);
    }
}
