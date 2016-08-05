package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by akshay.us on 3/15/16.
 */
public class BucketizerBridgeTest extends SparkTestBase {
    @Test
    public void bucketizerTest() {
        double[] validData = {-0.5, -0.3, 0.0, 0.2};
        double[] expectedBuckets = {0.0, 0.0, 1.0, 1.0};
        double[] splits = {-0.5, 0.0, 0.5};

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("feature", DataTypes.DoubleType, false, Metadata.empty())
        });
        List<Row> trainingData = Arrays.asList(
                cr(0, validData[0]),
                cr(1, validData[1]),
                cr(2, validData[2]),
                cr(3, validData[3]));

        Dataset<Row> df = spark.createDataFrame(trainingData, schema);

        Bucketizer sparkModel = new Bucketizer()
                .setInputCol("feature")
                .setOutputCol("result")
                .setSplits(splits);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        List<Row> sparkOutput = sparkModel.transform(df).orderBy("id").select("id", "feature", "result").collectAsList();

        for (Row r : sparkOutput) {
            double input = r.getDouble(1);
            double sparkOp = r.getDouble(2);

            Map<String, Object> data = new HashMap<String, Object>();
            data.put(sparkModel.getInputCol(), input);
            transformer.transform(data);
            double transformedInput = (double) data.get(sparkModel.getOutputCol());

            assertTrue((transformedInput >= 0) && (transformedInput <= 1));
            assertEquals(transformedInput, sparkOp, 0.01);
            assertEquals(transformedInput, expectedBuckets[r.getInt(0)], 0.01);
        }
    }
}
