package com.flipkart.fdp.ml.modelinfo.adapter;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogisticRegressionModelInfoInfoAdapter
        implements ModelInfoAdapter<LogisticRegressionModel, LogisticRegressionModelInfo> {
    private static final Logger LOG =
            LoggerFactory.getLogger(LogisticRegressionModelInfoInfoAdapter.class);

    @Override
    public LogisticRegressionModelInfo transform(LogisticRegressionModel sparkLRModel) {
        LogisticRegressionModelInfo logisticRegressionModelInfo = new LogisticRegressionModelInfo();
        logisticRegressionModelInfo.weights = sparkLRModel.weights().toArray();
        logisticRegressionModelInfo.intercept = sparkLRModel.intercept();
        logisticRegressionModelInfo.numClasses = sparkLRModel.numClasses();
        logisticRegressionModelInfo.numFeatures = sparkLRModel.numFeatures();
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
