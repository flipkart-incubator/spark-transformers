package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.mllib.classification.LogisticRegressionModel;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link LogisticRegressionModel} in MlLib to  {@link LogisticRegressionModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
public class LogisticRegressionModelInfoAdapter
        extends AbstractModelInfoAdapter<LogisticRegressionModel, LogisticRegressionModelInfo> {

    @Override
    public LogisticRegressionModelInfo getModelInfo(final LogisticRegressionModel sparkLRModel) {
        final LogisticRegressionModelInfo logisticRegressionModelInfo = new LogisticRegressionModelInfo();
        logisticRegressionModelInfo.setWeights(sparkLRModel.weights().toArray());
        logisticRegressionModelInfo.setIntercept(sparkLRModel.intercept());
        logisticRegressionModelInfo.setNumClasses(sparkLRModel.numClasses());
        logisticRegressionModelInfo.setNumFeatures(sparkLRModel.numFeatures());
        logisticRegressionModelInfo.setThreshold((double) sparkLRModel.getThreshold().get());

        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add("features");
        logisticRegressionModelInfo.setInputKeys(inputKeys);

        Set<String> outputKeys = new LinkedHashSet<String>();
        outputKeys.add("prediction");
        outputKeys.add("probability");
        logisticRegressionModelInfo.setOutputKeys(outputKeys);

        return logisticRegressionModelInfo;
    }

    @Override
    public Class<LogisticRegressionModel> getSource() {
        return LogisticRegressionModel.class;
    }

    @Override
    public Class<LogisticRegressionModelInfo> getTarget() {
        return LogisticRegressionModelInfo.class;
    }

}
