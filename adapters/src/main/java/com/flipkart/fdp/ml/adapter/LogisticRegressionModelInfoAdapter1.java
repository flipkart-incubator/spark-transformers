package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.DataFrame;

/**
 * Transforms Spark's {@link LogisticRegressionModel} to  {@link LogisticRegressionModelInfo} object
 * that can be exported through {@link com.flipkart.fdp.ml.export.ModelExporter}
 */
@Slf4j
public class LogisticRegressionModelInfoAdapter1
        implements ModelInfoAdapter<LogisticRegressionModel, LogisticRegressionModelInfo> {

    @Override
    public LogisticRegressionModelInfo getModelInfo(final LogisticRegressionModel sparkLRModel, DataFrame df) {
        final LogisticRegressionModelInfo logisticRegressionModelInfo = new LogisticRegressionModelInfo();
        logisticRegressionModelInfo.setWeights(sparkLRModel.coefficients().toArray());
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
