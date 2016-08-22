package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class LogisticRegressionBridgeTest extends SparkTestBase {

    @Test
    public void testLogisticRegression() {
        //prepare data
        String datapath = "src/test/resources/binary_classification_test.libsvm";
        JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

        //Train model in spark
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(trainingData.rdd());

        //Export this model
        byte[] exportedModel = ModelExporter.export(lrmodel);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //validate predictions
        List<LabeledPoint> testPoints = trainingData.collect();
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = lrmodel.predict(v);

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("features", v.toArray());
            transformer.transform(data);
            double predicted = (double) data.get("prediction");

            assertEquals(actual, predicted, 0.01);
        }
    }
}
