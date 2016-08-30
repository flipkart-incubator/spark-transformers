package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.apache.spark.sql.types.DataTypes.*;
import static org.junit.Assert.assertEquals;

/**
 * Created by akshay.us on 3/21/16.
 */
public class PipelineBridgeTest extends SparkTestBase {

    @Test
    public void testPipeline() {
        // Prepare training documents, which are labeled.
        StructType schema = createStructType(new StructField[]{
                createStructField("id", LongType, false),
                createStructField("text", StringType, false),
                createStructField("label", DoubleType, false)
        });
        DataFrame trainingData = sqlContext.createDataFrame(Arrays.asList(
                cr(0L, "a b c d e spark", 1.0),
                cr(1L, "b d", 0.0),
                cr(2L, "spark f g h", 1.0),
                cr(3L, "hadoop mapreduce", 0.0)
        ), schema);

        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and LogisticRegression.
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("text")
                .setOutputCol("words")
                .setPattern("\\s")
                .setGaps(true)
                .setToLowercase(false);

        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.01);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{tokenizer, hashingTF, lr});

        // Fit the pipeline to training documents.
        PipelineModel sparkPipelineModel = pipeline.fit(trainingData);


        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkPipelineModel, trainingData);
        System.out.println(new String(exportedModel));

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //prepare test data
        StructType testSchema = createStructType(new StructField[]{
                createStructField("id", LongType, false),
                createStructField("text", StringType, false),
        });
        DataFrame testData = sqlContext.createDataFrame(Arrays.asList(
                cr(4L, "spark i j k"),
                cr(5L, "l m n"),
                cr(6L, "mapreduce spark"),
                cr(7L, "apache hadoop")
        ), testSchema);

        //verify that predictions for spark pipeline and exported pipeline are the same
        Row[] predictions = sparkPipelineModel.transform(testData).select("id", "text", "probability", "prediction").collect();
        for (Row r : predictions) {
            System.out.println(r);
            double sparkPipelineOp = r.getDouble(3);
            Map<String, Object> data = new HashMap<String, Object>();
            data.put("text", r.getString(1));
            transformer.transform(data);
            double exportedPipelineOp = (double) data.get("prediction");
            double exportedPipelineProb = (double) data.get("probability");
            assertEquals(sparkPipelineOp, exportedPipelineOp, EPSILON);
        }
    }
}
