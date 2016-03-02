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

import java.util.List;

import static org.junit.Assert.assertEquals;

public class LogisticRegressionBridgeTest extends SparkTestBase {

    @Test
    public void testLogisticRegression() {
        //prepare data
        String datapath = "src/test/resources/binary_classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();

        //Train model in spark
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());

        //Export this model
        byte[] exportedModel = ModelExporter.export(lrmodel);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //validate predictions
        lrmodel.clearThreshold();
        List<LabeledPoint> testPoints = data.collect();
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = lrmodel.predict(v);
            double predicted = transformer.transform(v.toArray());
            //System.out.println(actual + "  -- " + predicted);
            assertEquals(actual, predicted, 0.01);
        }
    }
}
