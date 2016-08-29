package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by akshay.us on 8/29/16.
 */
public class DecisionTreeRegressionModelBridgeTest extends SparkTestBase {


    @Test
    public void testDecisionTreeRegression() {
        // Load the data stored in LIBSVM format as a DataFrame.
        DataFrame data = sqlContext.read().format("libsvm").load("src/test/resources/regression_test.libsvm");

        // Split the data into training and test sets (30% held out for testing)
        DataFrame[] splits = data.randomSplit(new double[]{0.7, 0.3});
        DataFrame trainingData = splits[0];
        DataFrame testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeRegressionModel regressionModel = new DecisionTreeRegressor()
                .setFeaturesCol("features").fit(trainingData);

        byte[] exportedModel = ModelExporter.export(regressionModel, null);

        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        Row[] sparkOutput = regressionModel.transform(testData).select("features", "prediction").collect();

        //compare predictions
        for (Row row : sparkOutput) {
            Vector v = (Vector) row.get(0);
            double actual = row.getDouble(1);

            Map<String, Object> inputData = new HashMap<String, Object>();
            inputData.put(transformer.getInputKeys().iterator().next(), v.toArray());
            transformer.transform(inputData);
            double predicted = (double) inputData.get(transformer.getOutputKeys().iterator().next());

            System.out.println(actual + ", "+predicted);
            assertEquals(actual, predicted, 0.01);
        }
    }


    @Test
    public void testDecisionTreeRegressionWithPipeline() {
        // Load the data stored in LIBSVM format as a DataFrame.
        DataFrame data = sqlContext.read().format("libsvm").load("src/test/resources/regression_test.libsvm");

        // Split the data into training and test sets (30% held out for testing)
        DataFrame[] splits = data.randomSplit(new double[]{0.7, 0.3});
        DataFrame trainingData = splits[0];
        DataFrame testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeRegressor dt = new DecisionTreeRegressor()
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{dt});

        // Train model.  This also runs the indexer.
        PipelineModel sparkPipeline = pipeline.fit(trainingData);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkPipeline, null);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        Row[] sparkOutput = sparkPipeline.transform(testData).select("features", "prediction").collect();

        //compare predictions
        for (Row row : sparkOutput) {
            Vector v = (Vector) row.get(0);
            double actual = row.getDouble(1);

            Map<String, Object> inputData = new HashMap<String, Object>();
            inputData.put(transformer.getInputKeys().iterator().next(), v.toArray());
            transformer.transform(inputData);
            double predicted = (double) inputData.get(transformer.getOutputKeys().iterator().next());

            assertEquals(actual, predicted, 0.01);
        }
    }

}
