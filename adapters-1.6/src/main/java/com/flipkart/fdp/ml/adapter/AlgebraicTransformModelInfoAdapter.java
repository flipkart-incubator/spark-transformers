package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.AlgebraicTransform;
import com.flipkart.fdp.ml.modelinfo.AlgebraicTransformModelInfo;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
public class AlgebraicTransformModelInfoAdapter extends AbstractModelInfoAdapter<AlgebraicTransform, AlgebraicTransformModelInfo> {
    @Override
    public AlgebraicTransformModelInfo getModelInfo(final AlgebraicTransform from, DataFrame df) {
        AlgebraicTransformModelInfo modelInfo = new AlgebraicTransformModelInfo();
        modelInfo.setCoefficients(from.getCoefficients());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);
        return modelInfo;
    }

    @Override
    public Class<AlgebraicTransform> getSource() {
        return AlgebraicTransform.class;
    }

    @Override
    public Class<AlgebraicTransformModelInfo> getTarget() {
        return AlgebraicTransformModelInfo.class;
    }
}
