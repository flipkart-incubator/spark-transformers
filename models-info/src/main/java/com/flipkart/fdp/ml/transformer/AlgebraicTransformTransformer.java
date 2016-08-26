package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.AlgebraicTransformModelInfo;

import java.util.Map;
import java.util.Set;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
public class AlgebraicTransformTransformer implements Transformer {
    private final AlgebraicTransformModelInfo modelInfo;

    public AlgebraicTransformTransformer(final AlgebraicTransformModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public double predict(final double input) {
        double[] coeff = modelInfo.getCoefficients();
        if(coeff.length == 0){
            return 0.0;
        }
        else{
            double sum = coeff[0];
            double mul = input;
            for(int i = 1; i < coeff.length; i++){
                sum = sum + (coeff[i] * mul);
                mul = mul * input;
            }
            return sum;
        }
    }

    @Override
    public void transform(Map<String, Object> input) {
        double inp = (double) input.get(modelInfo.getInputKeys().iterator().next());
        input.put(modelInfo.getOutputKeys().iterator().next(), predict(inp));
    }

    @Override
    public Set<String> getInputKeys() {
        return modelInfo.getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return modelInfo.getOutputKeys();
    }

}
