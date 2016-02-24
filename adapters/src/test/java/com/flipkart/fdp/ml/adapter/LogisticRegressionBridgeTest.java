package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import com.flipkart.fdp.ml.predictors.LogisticRegressionPredictor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by akshay.us on 2/24/16.
 */
public class LogisticRegressionBridgeTest extends SparkTestBase{

    @Test
    public void testLogisticRegression() {
        String datapath = "src/test/resources/binary_classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());
        LogisticRegressionModelInfoInfoAdapter logisticRegressionBridgeIn =
                new LogisticRegressionModelInfoInfoAdapter();

        LogisticRegressionModelInfo logisticRegressionModelInfo =
                logisticRegressionBridgeIn.transform(lrmodel);
        LogisticRegressionPredictor predictor =
                new LogisticRegressionPredictor(logisticRegressionModelInfo);
        lrmodel.clearThreshold();
        List<LabeledPoint> testPoints = data.collect();
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = lrmodel.predict(v);
            double predicted = predictor.predict(v.toArray());
            System.out.println(actual + "  -- " + predicted);
            assertEquals(actual, predicted, 0.01);
        }
    }
}
