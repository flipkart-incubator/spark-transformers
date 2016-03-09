package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.StringIndexerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents information for StringIndexer model
 */
@Data
public class StringIndexerModelInfo implements ModelInfo {

    private Map<String, Double> labelToIndex = new HashMap<String, Double>();

    /**
     * @return an corresponding {@link StringIndexerTransformer} for this model info
     */
    @Override
    public Transformer getTransformer() {
        return new StringIndexerTransformer(this);
    }
}
