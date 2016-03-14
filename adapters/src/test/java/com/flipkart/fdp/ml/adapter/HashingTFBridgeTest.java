package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.types.DataTypes.*;
import static org.junit.Assert.assertArrayEquals;

/**
 * Created by akshay.us on 3/8/16.
 */
public class HashingTFBridgeTest extends SparkTestBase {

    @Test
    public void testHashingTF() {
        //prepare data
        List<Row> data = Arrays.asList(
                RowFactory.create(1, 0.0, "Hi I heard about Spark"),
                RowFactory.create(2, 0.0, "I wish Java could use case classes"),
                RowFactory.create(3, 1.0, "Logistic regression models are neat")
        );
        StructType schema = new StructType(new StructField[]{
                createStructField("id", IntegerType, false),
                createStructField("label", DoubleType, false),
                createStructField("sentence", StringType, false),
        });

        DataFrame sentenceData = sqlContext.createDataFrame(data, schema);
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("sentence")
                .setOutputCol("words");
        DataFrame wordsData = tokenizer.transform(sentenceData);

        //train model in spark
        int numFeatures = 20;
        HashingTF sparkModel = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel, sentenceData);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        Row[] sparkOutput = sparkModel.transform(wordsData).orderBy("id").select("id", "sentence", "words", "rawFeatures").collect();
        for (Row row : sparkOutput) {
            String[] words = ((String) row.get(1)).toLowerCase().split(" ");
            double[] transformedOp = ArrayUtils.toPrimitive((Double[])transformer.transform(words));
            double[] sparkOp = ((Vector) row.get(3)).toArray();
            assertArrayEquals(transformedOp, sparkOp, 0.01);
        }
    }
}
