package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.DataFrame;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Transforms Spark's {@link StringIndexerModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class StringIndexerModelInfoAdapter extends AbstractModelInfoAdapter<StringIndexerModel, StringIndexerModelInfo> {

    @Override
    public StringIndexerModelInfo getModelInfo(final StringIndexerModel from, DataFrame df) {
        final String[] labels = from.labels();
        final Map<String, Double> labelToIndex = new HashMap<String, Double>();
        for (int i = 0; i < labels.length; i++) {
            labelToIndex.put(labels[i], (double) i);
        }
        final StringIndexerModelInfo modelInfo = new StringIndexerModelInfo();
        modelInfo.setLabelToIndex(labelToIndex);
        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);
        modelInfo.setOutputKey(from.getOutputCol());
        return modelInfo;
    }

    @Override
    public Class<StringIndexerModel> getSource() {
        return StringIndexerModel.class;
    }

    @Override
    public Class<StringIndexerModelInfo> getTarget() {
        return StringIndexerModelInfo.class;
    }
}
