package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.HashingTFModelInfo;
import org.apache.spark.ml.feature.HashingTF;

/**
 * Created by akshay.us on 3/4/16.
 */
public class HashingTFModelInfoAdapter implements ModelInfoAdapter<HashingTF, HashingTFModelInfo> {
    @Override
    public HashingTFModelInfo getModelInfo(HashingTF from) {
        HashingTFModelInfo modelInfo = new HashingTFModelInfo();
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
