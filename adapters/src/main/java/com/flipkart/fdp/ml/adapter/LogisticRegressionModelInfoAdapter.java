package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.mllib.classification.LogisticRegressionModel;

/**
 * Transforms Spark's {@link LogisticRegressionModel} in MlLib to  {@link com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
public class LogisticRegressionModelInfoAdapter
        implements ModelInfoAdapter<LogisticRegressionModel, LogisticRegressionModelInfo> {

    @Override
    public LogisticRegressionModelInfo getModelInfo(LogisticRegressionModel sparkLRModel) {
        LogisticRegressionModelInfo logisticRegressionModelInfo = new LogisticRegressionModelInfo();
        logisticRegressionModelInfo.setWeights(sparkLRModel.weights().toArray());
        logisticRegressionModelInfo.setIntercept(sparkLRModel.intercept());
        logisticRegressionModelInfo.setNumClasses(sparkLRModel.numClasses());
        logisticRegressionModelInfo.setNumFeatures(sparkLRModel.numFeatures());
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
