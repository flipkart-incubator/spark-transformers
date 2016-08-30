package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by akshay.us on 3/14/16.
 */
public class CountVectorizerBridgeTest extends SparkTestBase {

    @Test
    public void testCountVectorizer() {

        final List<List<String>> input = new ArrayList<>();
        input.add(Arrays.<String>asList("a", "b", "c"));
        input.add(Arrays.<String>asList("a", "b", "b", "c", "a"));

        //prepare data
        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create(1, input.get(0)),
                RowFactory.create(2, input.get(1))
        ));
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        DataFrame df = sqlContext.createDataFrame(jrdd, schema);

        //train model in spark
        CountVectorizerModel sparkModel = new CountVectorizer()
                .setInputCol("text")
                .setOutputCol("feature")
                .setVocabSize(3)
                .setMinDF(2)
                .fit(df);
        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel, df);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        Row[] sparkOutput = sparkModel.transform(df).orderBy("id").select("feature").collect();
        for (int i = 0; i < 2; i++) {
            Object[] words = input.get(i).toArray();

            Map<String, Object> data = new HashMap<String, Object>();
            data.put(sparkModel.getInputCol(), words);
            transformer.transform(data);
            double[] transformedOp = (double[]) data.get(sparkModel.getOutputCol());

            double[] sparkOp = ((Vector) sparkOutput[i].get(0)).toArray();
            assertArrayEquals(transformedOp, sparkOp, EPSILON);
        }
    }

}
