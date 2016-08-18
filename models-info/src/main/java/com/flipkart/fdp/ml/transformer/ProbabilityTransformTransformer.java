package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.AbstractModelInfo;
import com.flipkart.fdp.ml.modelinfo.ProbabilityTransformModelInfo;

import java.util.Map;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
public class ProbabilityTransformTransformer implements Transformer {
    private final ProbabilityTransformModelInfo modelInfo;

    public ProbabilityTransformTransformer(final ProbabilityTransformModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double predict(final double input) {
        double p1 = modelInfo.getActualClickProportion();
        double r1 = modelInfo.getUnderSampledClickProportion();
        double probIndex = modelInfo.getProbIndex();//not used because in the map LR only fills prob wrt positive class

        double encoding = (input *p1/r1) / ((input *p1/r1) + ((1-input) *(1-p1)/(1-r1)));
        return encoding;
    }

    @Override
    public void transform(Map<String, Object> input) {
        double inp = (double) input.get(modelInfo.getInputKeys().iterator().next());
        input.put(modelInfo.getOutputKey(), predict(inp));
    }
}
