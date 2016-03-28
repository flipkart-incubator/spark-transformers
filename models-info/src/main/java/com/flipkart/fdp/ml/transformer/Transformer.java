package com.flipkart.fdp.ml.transformer;

import java.util.Map;

/**
 * This interface represents a capability of a class to transform the input using a suitable model
 * representation captured in  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo}.
 */
public interface Transformer {

    /**
     * @param input values as map of (String, Object) for the transformation
     *              similar to the lines of a dataframe.
     */
    public void transform(Map<String, Object> input);

}
