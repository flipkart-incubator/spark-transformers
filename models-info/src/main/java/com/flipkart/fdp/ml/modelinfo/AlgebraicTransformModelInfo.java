package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.AlgebraicTransformTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
@Data
public class AlgebraicTransformModelInfo extends AbstractModelInfo{
    private double[] coefficients;

    @Override
    public Transformer getTransformer() {
        return new AlgebraicTransformTransformer(this);
    }
}
