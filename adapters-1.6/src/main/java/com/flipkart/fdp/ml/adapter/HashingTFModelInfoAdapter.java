package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link HashingTF} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class HashingTFModelInfoAdapter extends AbstractModelInfoAdapter<HashingTF, HashingTFModelInfo> {
    @Override
    public HashingTFModelInfo getModelInfo(final HashingTF from, DataFrame df) {
        final HashingTFModelInfo modelInfo = new HashingTFModelInfo();
        modelInfo.setNumFeatures(from.getNumFeatures());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<HashingTF> getSource() {
        return HashingTF.class;
    }

    @Override
    public Class<HashingTFModelInfo> getTarget() {
        return HashingTFModelInfo.class;
    }
}
