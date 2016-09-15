package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.FillNAValuesTransformer;
import com.flipkart.fdp.ml.modelinfo.FillNAValuesTransformerModelInfo;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms {@link FillNAValuesTransformer} to  {@link FillNAValuesTransformerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class FillNAValuesTransformerModelInfoAdapter extends AbstractModelInfoAdapter<FillNAValuesTransformer, FillNAValuesTransformerModelInfo> {

    @Override
    public FillNAValuesTransformerModelInfo getModelInfo(final FillNAValuesTransformer from, DataFrame df) {

        final FillNAValuesTransformerModelInfo modelInfo = new FillNAValuesTransformerModelInfo();
        modelInfo.setNaValuesMap(from.getNAValueMap());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.addAll(from.getNAValueMap().keySet());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.addAll(from.getNAValueMap().keySet());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class<FillNAValuesTransformer> getSource() {
        return FillNAValuesTransformer.class;
    }

    @Override
    public Class<FillNAValuesTransformerModelInfo> getTarget() {
        return FillNAValuesTransformerModelInfo.class;
    }
}
