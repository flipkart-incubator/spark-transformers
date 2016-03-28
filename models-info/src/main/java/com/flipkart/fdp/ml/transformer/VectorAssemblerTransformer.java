package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.VectorAssemblerModelInfo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Created by rohan.shetty on 28/03/16.
 */
public class VectorAssemblerTransformer implements Transformer {
    private final VectorAssemblerModelInfo modelInfo;

    public VectorAssemblerTransformer(final VectorAssemblerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    private double[] predict (Object[] inputs) {
        List<Double> output = new ArrayList<>();
        for (Object input : inputs) {
            if (input == null) {
                throw new RuntimeException("Values to assemble cannot be null");
            }
            else if (double.class.equals(input.getClass()) ||
                        Double.class.equals(input.getClass())) {
                output.add((double)input);
            }
            else if (double[].class.equals(input.getClass())) {
                for (Double val : (double[])input) {
                    output.add(val);
                }
            }
            else {
                throw new RuntimeException("Values to assemble cannot be of type: " + input.getClass().getCanonicalName());
            }
        }
        return output.stream().mapToDouble(Double::doubleValue).toArray();
    }

    @Override
    public void transform(Map<String, Object> input) {
        Object[] inputs = new Object[modelInfo.getInputKeys().size()];
        int i =0;
        for (String inputKey : modelInfo.getInputKeys()) {
            inputs[i++] = input.get(inputKey);
        }
        input.put(modelInfo.getOutputKey(), predict(inputs));
    }
}
