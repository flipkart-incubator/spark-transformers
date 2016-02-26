package com.flipkart.fdp.ml.transformer;

/**
 * This interface represents a capability of a class to transform the input using a suitable model
 * representation captured in  {@link com.flipkart.fdp.ml.modelinfo.ModelInfo}.
 * */
public interface Transformer {

    /**
     * @param input values as a array of double for the transformation
     * @return prediction / transformed input
     * */
    double transform(double[] input);
}
