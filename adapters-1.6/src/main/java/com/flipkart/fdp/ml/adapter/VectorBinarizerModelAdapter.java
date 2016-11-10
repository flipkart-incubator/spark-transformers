package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.VectorBinarizerModelInfo;
import org.apache.spark.ml.feature.VectorBinarizer;

import org.apache.spark.sql.DataFrame;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;


/**
 * Transforms  {@link org.apache.spark.ml.feature.VectorBinarizer} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.VectorBinarizerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 * <p>
 * Created by karan.verma on 9/11/16.
 */

public class VectorBinarizerModelAdapter extends AbstractModelInfoAdapter<VectorBinarizer, VectorBinarizerModelInfo>  {
    @Override
    VectorBinarizerModelInfo getModelInfo(VectorBinarizer from, DataFrame df) {

        VectorBinarizerModelInfo vectorBinarizerModelInfo = new VectorBinarizerModelInfo();

        vectorBinarizerModelInfo.setInputKeys(new LinkedHashSet<>(Arrays.asList(from.getInputCol())));

        Set<String> outputKeys = new LinkedHashSet<String>();

        outputKeys.add(from.getOutputCol());
        vectorBinarizerModelInfo.setOutputKeys(outputKeys);
        vectorBinarizerModelInfo.setThreshold(from.getThreshold());

        return vectorBinarizerModelInfo;
    }

    @Override
    public Class<VectorBinarizer> getSource() {
        return VectorBinarizer.class;
    }

    @Override
    public Class<VectorBinarizerModelInfo> getTarget() {
        return VectorBinarizerModelInfo.class;
    }
}
