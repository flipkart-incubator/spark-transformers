package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.Log1PScaler;
import com.flipkart.fdp.ml.modelinfo.Log1PScalerModelInfo;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms {@link Log1PScaler} in MlLib to  {@link Log1PScalerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class Log1PScalerModelInfoAdapter extends AbstractModelInfoAdapter<Log1PScaler, Log1PScalerModelInfo> {

    @Override
    public Log1PScalerModelInfo getModelInfo(final Log1PScaler from, DataFrame df) {
        Log1PScalerModelInfo modelInfo = new Log1PScalerModelInfo();

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<Log1PScaler> getSource() {
        return Log1PScaler.class;
    }

    @Override
    public Class<Log1PScalerModelInfo> getTarget() {
        return Log1PScalerModelInfo.class;
    }
}
