package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.Transformer;

import java.io.Serializable;

/**
 * This interface represents information of a model. The implementors of this class should capture
 * the information(coefficients) of a model and the corresponding {@link Transformer} would use that
 * information for prediction/transformation
 */
public interface ModelInfo extends Serializable {

    /**
     * @return {@link Transformer} that will use the information(coefficients) of this model
     * to transform input
     */
    Transformer getTransformer();
}
