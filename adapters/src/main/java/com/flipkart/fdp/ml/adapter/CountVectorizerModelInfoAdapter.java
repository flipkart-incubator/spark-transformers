package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.CountVectorizerModelInfo;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link CountVectorizerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.CountVectorizerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class CountVectorizerModelInfoAdapter extends AbstractModelInfoAdapter<CountVectorizerModel, CountVectorizerModelInfo> {
    @Override
    public CountVectorizerModelInfo getModelInfo(final CountVectorizerModel from, final DataFrame df) {
        final CountVectorizerModelInfo modelInfo = new CountVectorizerModelInfo();
        modelInfo.setMinTF(from.getMinTF());
        modelInfo.setVocabSize(from.getVocabSize());
        modelInfo.setVocabulary(from.vocabulary());

        return modelInfo;
    }

    @Override
    public Class<CountVectorizerModel> getSource() {
        return CountVectorizerModel.class;
    }

    @Override
    public Class<CountVectorizerModelInfo> getTarget() {
        return CountVectorizerModelInfo.class;
    }
}
