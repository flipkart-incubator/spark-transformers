package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.Transformer;

/**
 * This is just a marker interface. The implementors of this class represent information on a model
 */

public interface ModelInfo {

    Transformer getTransformer();
}
