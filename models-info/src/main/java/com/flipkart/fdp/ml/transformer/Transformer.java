package com.flipkart.fdp.ml.transformer;

/**
 * This interface represents a capability of a class to transform the output of a model.
 * The model type and info is captured in the template parameter.
 */

public interface Transformer {
    double transform(double[] input);
}
