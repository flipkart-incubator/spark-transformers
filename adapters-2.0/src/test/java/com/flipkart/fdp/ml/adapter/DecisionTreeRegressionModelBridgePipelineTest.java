package com.flipkart.fdp.ml.adapter;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.junit.Test;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;

/**
 * 
 * @author harshit.pandey
 *
 */
public class DecisionTreeRegressionModelBridgePipelineTest extends SparkTestBase {


    @Test
    public void testDecisionTreeRegressionPrediction() {
        // Load the data stored in LIBSVM format as a DataFrame.
    	String datapath = "src/test/resources/regression_test.libsvm";
    	
    	Dataset<Row> data = spark.read().format("libsvm").load(datapath);


        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        StringIndexer indexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("labelIndex").setHandleInvalid("skip");
        
		DecisionTreeRegressor regressionModel =
		        new DecisionTreeRegressor().setLabelCol("labelIndex").setFeaturesCol("features");
		
		Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{indexer, regressionModel});

		PipelineModel sparkPipeline = pipeline.fit(trainingData);
		
        byte[] exportedModel = ModelExporter.export(sparkPipeline);

        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        List<Row> output = sparkPipeline.transform(testData).select("features", "prediction", "label").collectAsList();

        //compare predictions
        for (Row row : output) {
        	Map<String, Object> data_ = new HashMap<>();
            data_.put("features", ((SparseVector) row.get(0)).toArray());
            data_.put("label", (row.get(2)).toString());
            transformer.transform(data_);
            System.out.println(data_);
            System.out.println(data_.get("prediction"));
            assertEquals((double)data_.get("prediction"), (double)row.get(1), EPSILON);
        }
    }

}
