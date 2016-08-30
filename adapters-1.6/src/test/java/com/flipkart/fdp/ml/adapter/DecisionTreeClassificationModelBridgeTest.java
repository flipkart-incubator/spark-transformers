package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.DecisionTreeTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by akshay.us on 8/29/16.
 */
public class DecisionTreeClassificationModelBridgeTest extends SparkTestBase {


    @Test
    public void testDecisionTreeClassificationRawPrediction() {
        // Load the data stored in LIBSVM format as a DataFrame.
        DataFrame data = sqlContext.read().format("libsvm").load("src/test/resources/classification_test.libsvm");

        StringIndexerModel stringIndexerModel = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("labelIndex")
                .fit(data);

        data = stringIndexerModel.transform(data);

        // Split the data into training and test sets (30% held out for testing)
        DataFrame[] splits = data.randomSplit(new double[]{0.7, 0.3});
        DataFrame trainingData = splits[0];
        DataFrame testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeClassificationModel classificationModel = new DecisionTreeClassifier()
                .setLabelCol("labelIndex")
                .setFeaturesCol("features")
                .setRawPredictionCol("rawPrediction")
                .setPredictionCol("prediction")
                .fit(trainingData);

        byte[] exportedModel = ModelExporter.export(classificationModel, null);

        Transformer transformer = (DecisionTreeTransformer) ModelImporter.importAndGetTransformer(exportedModel);

        Row[] sparkOutput = classificationModel.transform(testData).select("features", "prediction", "rawPrediction").collect();

        //compare predictions
        for (Row row : sparkOutput) {
            Vector inp = (Vector) row.get(0);
            double actual = row.getDouble(1);
            double[] actualRaw = ((Vector) row.get(2)).toArray();

            Map<String, Object> inputData = new HashMap<>();
            inputData.put(transformer.getInputKeys().iterator().next(), inp.toArray());
            transformer.transform(inputData);
            double predicted = (double) inputData.get(transformer.getOutputKeys().iterator().next());
            double[] rawPrediction = (double[]) inputData.get("rawPrediction");

            assertEquals(actual, predicted, 0.01);
            assertArrayEquals(actualRaw, rawPrediction, 0.01);
        }
    }

    @Test
    public void testDecisionTreeClassificationWithPipeline() {
        // Load the data stored in LIBSVM format as a DataFrame.
        DataFrame data = sqlContext.read().format("libsvm").load("src/test/resources/classification_test.libsvm");

        // Split the data into training and test sets (30% held out for testing)
        DataFrame[] splits = data.randomSplit(new double[]{0.7, 0.3});
        DataFrame trainingData = splits[0];
        DataFrame testData = splits[1];

        StringIndexer indexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("labelIndex");

        // Train a DecisionTree model.
        DecisionTreeClassifier classificationModel = new DecisionTreeClassifier()
                .setLabelCol("labelIndex")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{indexer, classificationModel});

        // Train model.  This also runs the indexer.
        PipelineModel sparkPipeline = pipeline.fit(trainingData);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkPipeline, null);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        Row[] sparkOutput = sparkPipeline.transform(testData).select("label", "features", "prediction").collect();

        //compare predictions
        for (Row row : sparkOutput) {
            Vector v = (Vector) row.get(1);
            double actual = row.getDouble(2);

            Map<String, Object> inputData = new HashMap<String, Object>();
            inputData.put("features", v.toArray());
            inputData.put("label", row.get(0).toString());
            transformer.transform(inputData);
            double predicted = (double) inputData.get("prediction");

            assertEquals(actual, predicted, 0.01);
        }
    }

}
