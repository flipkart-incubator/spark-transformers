package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.IfZeroVector;
import com.flipkart.fdp.ml.modelinfo.IfZeroVectorModelInfo;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms {@link IfZeroVector} to {@link IfZeroVectorModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class IfZeroVectorModelInfoAdapter extends AbstractModelInfoAdapter<IfZeroVector, IfZeroVectorModelInfo> {

    @Override
    public IfZeroVectorModelInfo getModelInfo(final IfZeroVector from, DataFrame df) {
        IfZeroVectorModelInfo modelInfo = new IfZeroVectorModelInfo();

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        modelInfo.setThenSetValue(from.getThenSetValue());
        modelInfo.setElseSetCol(from.getElseSetCol());

        return modelInfo;
    }

    @Override
    public Class<IfZeroVector> getSource() {
        return IfZeroVector.class;
    }

    @Override
    public Class<IfZeroVectorModelInfo> getTarget() {
        return IfZeroVectorModelInfo.class;
    }
}
