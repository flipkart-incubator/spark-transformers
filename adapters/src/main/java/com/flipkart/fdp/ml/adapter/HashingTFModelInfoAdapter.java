package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo;
import org.apache.spark.ml.feature.HashingTF;

/**
 * Transforms Spark's {@link HashingTF} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class HashingTFModelInfoAdapter implements ModelInfoAdapter<HashingTF, HashingTFModelInfo> {
    @Override
    public HashingTFModelInfo getModelInfo(final HashingTF from) {
        final HashingTFModelInfo modelInfo = new HashingTFModelInfo();
        modelInfo.setNumFeatures(from.getNumFeatures());
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
