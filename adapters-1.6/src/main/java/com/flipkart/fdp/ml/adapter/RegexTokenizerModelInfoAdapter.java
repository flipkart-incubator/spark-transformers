package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.RegexTokenizerModelInfo;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.DataFrame;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link RegexTokenizer} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.RegexTokenizerModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
public class RegexTokenizerModelInfoAdapter extends AbstractModelInfoAdapter<RegexTokenizer, RegexTokenizerModelInfo> {

    @Override
    public RegexTokenizerModelInfo getModelInfo(final RegexTokenizer from, final DataFrame df) {
        final RegexTokenizerModelInfo modelInfo = new RegexTokenizerModelInfo();
        modelInfo.setMinTokenLength(from.getMinTokenLength());
        modelInfo.setGaps(from.getGaps());
        modelInfo.setPattern(from.getPattern());
        modelInfo.setToLowercase(from.getToLowercase());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(from.getInputCol());
        modelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add(from.getOutputCol());
        modelInfo.setOutputKeys(outputKeys);

        return modelInfo;
    }

    @Override
    public Class getSource() {
        return RegexTokenizer.class;
    }

    @Override
    public Class getTarget() {
        return RegexTokenizerModelInfo.class;
    }
}
