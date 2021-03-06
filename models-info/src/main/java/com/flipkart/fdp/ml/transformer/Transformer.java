package com.flipkart.fdp.ml.transformer;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

/**
 * This interface represents a capability of a class to transform the input using a suitable model
 * representation captured in  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo}.
 */
public interface Transformer extends Serializable {

    /**
     * @param input values as map of (String, Object) for the transformation
     *              similar to the lines of a dataframe.
     */
    public void transform(Map<String, Object> input);

    public Set<String> getInputKeys();

    public Set<String> getOutputKeys();
}
