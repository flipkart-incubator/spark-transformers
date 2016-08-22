package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by rohan.shetty on 28/03/16.
 */

public class VectorAssemblerBridgeTest extends SparkTestBase {

    @Test
    public void testVectorAssembler() {
        // prepare data

        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create(0d, 1d, new DenseVector(new double[]{2d, 3d})),
                RowFactory.create(1d, 2d, new DenseVector(new double[]{3d, 4d})),
                RowFactory.create(2d, 3d, new DenseVector(new double[]{4d, 5d})),
                RowFactory.create(3d, 4d, new DenseVector(new double[]{5d, 6d})),
                RowFactory.create(4d, 5d, new DenseVector(new double[]{6d, 7d}))
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("value1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("vector1", new VectorUDT(), false, Metadata.empty())
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"value1", "vector1"})
                .setOutputCol("feature");


        //Export this model
        byte[] exportedModel = ModelExporter.export(vectorAssembler, null);

        String exportedModelJson = new String(exportedModel);
        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        //compare predictions
        Row[] sparkOutput = vectorAssembler.transform(df).orderBy("id").select("id", "value1", "vector1", "feature").collect();
        for (Row row : sparkOutput) {

            Map<String, Object> data = new HashMap<>();
            data.put(vectorAssembler.getInputCols()[0], row.get(1));
            data.put(vectorAssembler.getInputCols()[1], ((DenseVector) row.get(2)).toArray());
            transformer.transform(data);
            double[] output = (double[]) data.get(vectorAssembler.getOutputCol());
            assertArrayEquals(output, ((DenseVector) row.get(3)).toArray(), 0d);
        }
    }
}

