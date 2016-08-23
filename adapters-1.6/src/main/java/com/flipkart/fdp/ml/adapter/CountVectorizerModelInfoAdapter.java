package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.CountVectorizerModelInfo;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link CountVectorizerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.CountVectorizerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class CountVectorizerModelInfoAdapter extends AbstractModelInfoAdapter<CountVectorizerModel, CountVectorizerModelInfo> {
    @Override
    public CountVectorizerModelInfo getModelInfo(final CountVectorizerModel from, final DataFrame df) {
        final CountVectorizerModelInfo modelInfo = new CountVectorizerModelInfo();
        modelInfo.setMinTF(from.getMinTF());
        modelInfo.setVocabulary(from.vocabulary());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

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
