package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.classification.LogisticRegressionModel;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Transforms Spark's {@link LogisticRegressionModel} to  {@link LogisticRegressionModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
public class LogisticRegressionModelInfoAdapter1
        extends AbstractModelInfoAdapter<LogisticRegressionModel, LogisticRegressionModelInfo> {

    @Override
    public LogisticRegressionModelInfo getModelInfo(final LogisticRegressionModel sparkLRModel) {
        final LogisticRegressionModelInfo logisticRegressionModelInfo = new LogisticRegressionModelInfo();
        logisticRegressionModelInfo.setWeights(sparkLRModel.coefficients().toArray());
        logisticRegressionModelInfo.setIntercept(sparkLRModel.intercept());
        logisticRegressionModelInfo.setNumClasses(sparkLRModel.numClasses());
        logisticRegressionModelInfo.setNumFeatures(sparkLRModel.numFeatures());
        logisticRegressionModelInfo.setThreshold(sparkLRModel.getThreshold());
        Set<String> inputKeys = new LinkedHashSet<String>();
        inputKeys.add(sparkLRModel.getFeaturesCol());
        logisticRegressionModelInfo.setInputKeys(inputKeys);
        logisticRegressionModelInfo.setOutputKey(sparkLRModel.getPredictionCol());
        logisticRegressionModelInfo.setProbabilityKey(sparkLRModel.getProbabilityCol());
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
