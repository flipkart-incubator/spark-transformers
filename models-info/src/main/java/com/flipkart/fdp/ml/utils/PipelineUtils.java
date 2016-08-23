package com.flipkart.fdp.ml.utils;

import com.flipkart.fdp.ml.transformer.Transformer;

import java.util.HashSet;
import java.util.Set;

/**
 * Utility to extract input columns for a pipeline
 */
public class PipelineUtils {
    public static Set<String> extractRequiredInputColumns(Transformer[] transformers) {
        Set<String> inputColumns = new HashSet<>();

        //Add inputs for each transformer in the input set
        for(Transformer t : transformers) {
            inputColumns.addAll(t.getInputKeys());
        }

        //remove non modifying columns of each transformer
        for(Transformer t : transformers) {
            //calculate set difference Set(outputs) - Set(inputs)
            Set<String> setDifference = new HashSet<>(t.getOutputKeys());
            setDifference.removeAll(t.getInputKeys());

            inputColumns.removeAll(setDifference);
        }

        //Not handled cases where a transformer replaces/modifies any column that is not its input column.
        return inputColumns;
    }
}
