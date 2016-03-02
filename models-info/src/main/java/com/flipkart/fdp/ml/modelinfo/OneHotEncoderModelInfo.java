package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Created by akshay.us on 3/3/16.
 */
@Data
public class OneHotEncoderModelInfo implements ModelInfo {

    private boolean shouldDropLast;
    private int numTypes;

    @Override
    public Transformer getTransformer() {
        return null;
    }
}
