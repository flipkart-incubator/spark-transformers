package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;

public class MinMaxScalerBridgeTest extends SparkTestBase {

    private final double data[][] = {{1, 0, Long.MIN_VALUE},
            {2, 0, 0},
            {3, 0, Long.MAX_VALUE},
            {1.0, 0, 0}};

    private final double expected[][] = {{-5, 0, -5},
            {0, 0, 0},
            {5, 0, 5},
            {-2.5, 0, 0}};


    @Test
    public void testMinMaxScaler() {
        //prepare data
        JavaRDD<Row> jrdd = jsc.parallelize(Arrays.asList(
                RowFactory.create(1.0, Vectors.dense(data[0])),
                RowFactory.create(2.0, Vectors.dense(data[1])),
                RowFactory.create(3.0, Vectors.dense(data[2])),
                RowFactory.create(4.0, Vectors.dense(data[3]))
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(jrdd, schema);

        //train model in spark
        MinMaxScalerModel sparkModel = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaled")
                .setMin(-5)
                .setMax(5)
                .fit(df);


        //Export model, import it back and get transformer
        byte[] exportedModel = ModelExporter.export(sparkModel);
        final Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        List<Row> sparkOutput = sparkModel.transform(df).orderBy("label").select("features", "scaled").collectAsList();
        assertCorrectness(sparkOutput, expected, transformer);
    }

    private void assertCorrectness(List<Row> sparkOutput, double[][] expected, Transformer transformer) {
        for (int i = 0; i < 3; i++) {
            double[] input = ((Vector) sparkOutput.get(i).get(0)).toArray();

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("features", input);
            transformer.transform(data);
            double[] transformedOp = (double[]) data.get("scaled");

            double[] sparkOp = ((Vector) sparkOutput.get(i).get(1)).toArray();
            assertArrayEquals(transformedOp, sparkOp, 0.01);
            assertArrayEquals(transformedOp, expected[i], 0.01);
        }
    }
}
