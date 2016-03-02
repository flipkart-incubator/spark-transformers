package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.StringIndexerModelInfo;
import org.apache.spark.ml.feature.StringIndexerModel;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by akshay.us on 3/2/16.
 */
public class StringIndexerModelInfoAdapter implements ModelInfoAdapter<StringIndexerModel, StringIndexerModelInfo> {

    @Override
    public StringIndexerModelInfo getModelInfo(StringIndexerModel from) {
        String[] labels = from.labels();
        Map<String, Double> labelToIndex = new HashMap<String, Double>();
        for( int i =0; i < labels.length; i++) {
            labelToIndex.put(labels[i], (double)i);
        }
        StringIndexerModelInfo modelInfo = new StringIndexerModelInfo();
        modelInfo.setLabelToIndex(labelToIndex);
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
