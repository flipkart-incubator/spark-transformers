package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.ProbabilityTransformModelInfo;
import com.flipkart.fdp.ml.ProbabilityTransformModel;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
public class ProbabilityTransformModelInfoAdapter extends AbstractModelInfoAdapter<ProbabilityTransformModel, ProbabilityTransformModelInfo> {
    @Override
    public ProbabilityTransformModelInfo getModelInfo(final ProbabilityTransformModel from, DataFrame df) {
        ProbabilityTransformModelInfo modelInfo = new ProbabilityTransformModelInfo();

        modelInfo.setActualClickProportion(from.getActualClickProportion());
        modelInfo.setUnderSampledClickProportion(from.getUnderSampledClickProportion());
        modelInfo.setProbIndex(from.getProbIndex());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);
        return modelInfo;
    }

    @Override
    public Class<ProbabilityTransformModel> getSource() {
        return ProbabilityTransformModel.class;
    }

    @Override
    public Class<ProbabilityTransformModelInfo> getTarget() {
        return ProbabilityTransformModelInfo.class;
    }
}
