package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.VectorAssemblerModelInfo;

import java.util.Map;
import java.util.stream.Stream;

/**
 *
 * Transforms input/ predicts for a Vector assembler model representation
 * captured by  {@link com.flipkart.fdp.ml.modelinfo.VectorAssemblerModelInfo}.
 *
 * Created by rohan.shetty on 28/03/16.
 */
public class VectorAssemblerTransformer implements Transformer {
    private final VectorAssemblerModelInfo modelInfo;

    public VectorAssemblerTransformer(final VectorAssemblerModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    private double[] predict (Object[] inputs) {

        int vectorSize = Stream.of(inputs)
                .filter(o->isTypeDouble(o) || isTypeDoubleArray(o))
                .mapToInt(o -> isTypeDoubleArray(o) ? ((double[])o).length : 1)
                .sum();

        double[] output = new double[vectorSize];
        int i=0;
        for (Object input : inputs) {
            if (input == null) {
                throw new RuntimeException("Values to assemble cannot be null");
            }
            else if (isTypeDouble(input)) {
                output[i++] = (double) input;
            }
            else if (isTypeDoubleArray(input)) {
                for (double val : (double[])input) {
                    output[i++] = val;
                }
            }
            else {
                throw new RuntimeException("Values to assemble cannot be of type: " + input.getClass().getCanonicalName());
            }
        }
        return output;
    }

    private boolean isTypeDouble(Object o) {
        return o != null &&
                (double.class.equals(o.getClass()) ||
                        Double.class.equals(o.getClass()));
    }

    private boolean isTypeDoubleArray(Object o) {
        return o!=null && double[].class.equals(o.getClass());
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
