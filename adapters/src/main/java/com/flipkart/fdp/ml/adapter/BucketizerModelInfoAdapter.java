package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.BucketizerModelInfo;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link Bucketizer} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.BucketizerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class BucketizerModelInfoAdapter implements ModelInfoAdapter<Bucketizer, BucketizerModelInfo> {

    @Override
    public BucketizerModelInfo getModelInfo(final Bucketizer from, final DataFrame df) {
        final BucketizerModelInfo modelInfo = new BucketizerModelInfo();
        modelInfo.setSplits(from.getSplits());
        return modelInfo;
    }

    @Override
    public Class<Bucketizer> getSource() {
        return Bucketizer.class;
    }

    @Override
    public Class<BucketizerModelInfo> getTarget() {
        return BucketizerModelInfo.class;
    }
}
