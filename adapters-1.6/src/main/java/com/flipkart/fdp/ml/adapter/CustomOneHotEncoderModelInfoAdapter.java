package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.CustomOneHotEncoderModel;
import com.flipkart.fdp.ml.modelinfo.OneHotEncoderModelInfo;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Created by shubhranshu.shekhar on 21/06/16.
 */
public class CustomOneHotEncoderModelInfoAdapter extends AbstractModelInfoAdapter<CustomOneHotEncoderModel, OneHotEncoderModelInfo> {

    @Override
    public OneHotEncoderModelInfo getModelInfo(final CustomOneHotEncoderModel from, DataFrame df) {
        OneHotEncoderModelInfo modelInfo = new OneHotEncoderModelInfo();

        modelInfo.setNumTypes(from.vectorSize());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<CustomOneHotEncoderModel> getSource() {
        return CustomOneHotEncoderModel.class;
    }

    @Override
    public Class<OneHotEncoderModelInfo> getTarget() {
        return OneHotEncoderModelInfo.class;
    }
}
