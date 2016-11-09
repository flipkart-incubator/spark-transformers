package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.IfZeroVectorModelInfo;
import com.flipkart.fdp.ml.modelinfo.VectorAssemblerModelInfo;
import com.flipkart.fdp.ml.modelinfo.VectorBinarizerModelInfo;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorBinarizer;
import org.apache.spark.sql.DataFrame;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Created by karan.verma on 09/11/16.
 */
public class VectorBinarizerModelAdapter extends AbstractModelInfoAdapter<VectorBinarizer, VectorBinarizerModelInfo>  {
    @Override
    VectorBinarizerModelInfo getModelInfo(VectorBinarizer from, DataFrame df) {

        VectorBinarizerModelInfo vectorBinarizerModelInfo = new VectorBinarizerModelInfo();

        vectorBinarizerModelInfo.setInputKeys(new LinkedHashSet<>(Arrays.asList(from.getInputCol())));

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        vectorBinarizerModelInfo.setOutputKeys(outputKeys);

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
