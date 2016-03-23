package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.RegexTokenizerModelInfo;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.DataFrame;

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
