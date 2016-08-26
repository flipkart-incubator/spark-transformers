package com.flipkart.fdp.ml.modelinfo;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
import com.flipkart.fdp.ml.transformer.ProbabilityTransformTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

@Data
public class ProbabilityTransformModelInfo  extends AbstractModelInfo {
    private double actualClickProportion;
    private double underSampledClickProportion;
    private int probIndex;

    @Override
    public Transformer getTransformer() {
        return new ProbabilityTransformTransformer(this);
    }
}
