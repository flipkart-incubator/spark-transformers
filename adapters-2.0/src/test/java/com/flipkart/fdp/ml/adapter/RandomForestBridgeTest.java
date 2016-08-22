package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class RandomForestBridgeTest extends SparkTestBase {

    @Test
    public void testRandomForestBridgeClassification() {
        Integer numClasses = 7;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

        //prepare data
        String datapath = "src/test/resources/classification_test.libsvm";
        JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

        //Train model in scala
        RandomForestModel sparkModel = RandomForest
                .trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees,
                        featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //verify that predictions are the same
        List<LabeledPoint> testPoints = trainingData.take(10);
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = sparkModel.predict(v);

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("input", v.toArray());
            transformer.transform(data);
            double predicted = (double) data.get("output");

            assertEquals(actual, predicted, 0.01);
        }
    }

    @Test
    public void testRFRegression() {
        String datapath = "src/test/resources/regression_test.libsvm";
        testRegressor("variance", 3, 4, 32, "auto", 12345, datapath);
    }

    private void testRegressor(String impurity, int numTrees, int maxDepth, int maxBins,
                               String featureSubsetStrategy, int seed, String datapath) {

        //prepare data
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

        //Train model in spark
        RandomForestModel sparkModel = RandomForest
                .trainRegressor(trainingData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy,
                        impurity, maxDepth, maxBins, seed);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //verify that predictions are the same
        List<LabeledPoint> testPoints = trainingData.collect();
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = sparkModel.predict(v);

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("input", v.toArray());
            transformer.transform(data);
            double predicted = (double) data.get("output");

            assertEquals(actual, predicted, 0.01);
        }
    }
}
