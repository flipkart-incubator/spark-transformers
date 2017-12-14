package com.flipkart.fdp.ml.adapter;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.junit.Test;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.DecisionTreeTransformer;

/**
 * 
 * @author harshit.pandey
 *
 */
public class DecisionTreeClassificationModelBridgeTest extends SparkTestBase {


    @Test
    public void testDecisionTreeClassificationPrediction() {
        // Load the data stored in LIBSVM format as a DataFrame.
    	String datapath = "src/test/resources/classification_test.libsvm";
    	Dataset<Row> data = spark.read().format("libsvm").load(datapath);


        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeClassificationModel classifierModel = new DecisionTreeClassifier().fit(trainingData);
        trainingData.printSchema();
        
        List<Row> output = classifierModel.transform(testData).select("features", "prediction","rawPrediction").collectAsList();
        byte[] exportedModel = ModelExporter.export(classifierModel);

        DecisionTreeTransformer transformer = (DecisionTreeTransformer) ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        for (Row row : output) {
        	Map<String, Object> data_ = new HashMap<>();
        	double [] actualRawPrediction = ((DenseVector) row.get(2)).toArray();
            data_.put("features", ((SparseVector) row.get(0)).toArray());
            transformer.transform(data_);
            System.out.println(data_);
            System.out.println(data_.get("prediction"));
            assertEquals((double)data_.get("prediction"), (double)row.get(1), EPSILON);
            assertArrayEquals((double[]) data_.get("rawPrediction"), actualRawPrediction, EPSILON);
        }
    }
}
