package com.flipkart.fdp.ml.adapter;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
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
public class DecisionTreeRegressionModelBridgeTest extends SparkTestBase {


    @Test
    public void testDecisionTreeRegressionPrediction() {
        // Load the data stored in LIBSVM format as a DataFrame.
    	String datapath = "src/test/resources/regression_test.libsvm";
    	
    	Dataset<Row> data = spark.read().format("libsvm").load(datapath);


        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeRegressionModel regressionModel = new DecisionTreeRegressor().fit(trainingData);
        trainingData.printSchema();
        
        List<Row> output = regressionModel.transform(testData).select("features", "prediction").collectAsList();
        byte[] exportedModel = ModelExporter.export(regressionModel);

        DecisionTreeTransformer transformer = (DecisionTreeTransformer) ModelImporter.importAndGetTransformer(exportedModel);

        System.out.println(transformer);
        //compare predictions
        for (Row row : output) {
        	Map<String, Object> data_ = new HashMap<>();
            data_.put("features", ((SparseVector) row.get(0)).toArray());
            transformer.transform(data_);
            System.out.println(data_);
            System.out.println(data_.get("prediction"));
            assertEquals((double)data_.get("prediction"), (double)row.get(1), EPSILON);
        }
    }

}
