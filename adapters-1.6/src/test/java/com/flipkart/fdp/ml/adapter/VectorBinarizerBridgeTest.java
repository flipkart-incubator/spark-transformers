package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.VectorBinarizer;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.SparseVector;
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
 * Created by karan.verma on 09/11/16.
 */
public class VectorBinarizerBridgeTest extends SparkTestBase{

    @Test(expected=IllegalArgumentException.class)
    public void testVectorBinarizerNegativeThresholdValue() {
        // prepare data
        VectorBinarizer vectorBinarizer = new VectorBinarizer()
                .setInputCol("vector1")
                .setOutputCol("binarized")
                .setThreshold(-1d);
    }


    @Test
    public void testVectorBinarizerDense() {
        // prepare data

        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create(0d, 1d, new DenseVector(new double[]{-2d, -3d, -4d, -1d, 6d, -7d, 8d, 0d, 0d, 0d, 0d, 0d})),
                RowFactory.create(1d, 2d, new DenseVector(new double[]{4d, -5d, 6d, 7d, -8d, 9d, -10d, 0d, 0d, 0d, 0d, 0d})),
                RowFactory.create(2d, 3d, new DenseVector(new double[]{-5d, 6d, -8d, 9d, 10d, 11d, 12d, 0d, 0d, 0d, 0d, 0d}))
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("value1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("vector1", new VectorUDT(), false, Metadata.empty())
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);
        VectorBinarizer vectorBinarizer = new VectorBinarizer()
                .setInputCol("vector1")
                .setOutputCol("binarized")
                .setThreshold(2d);


        //Export this model
        byte[] exportedModel = ModelExporter.export(vectorBinarizer, df);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        //compare predictions
        Row[] sparkOutput = vectorBinarizer.transform(df).orderBy("id").select("id", "value1", "vector1", "binarized").collect();
        for (Row row : sparkOutput) {

            Map<String, Object> data = new HashMap<>();
            data.put(vectorBinarizer.getInputCol(), ((DenseVector) row.get(2)).toArray());
            transformer.transform(data);
            double[] output = (double[]) data.get(vectorBinarizer.getOutputCol());
            assertArrayEquals(output, ((DenseVector) row.get(3)).toArray(), 0d);
        }
    }

    @Test
    public void testVectorBinarizerSparse() {
        // prepare data

        int[] sparseArray1 = {5, 6, 11, 4, 7, 9, 8, 14, 13};
        double[] sparseArray1Values = {-5d, 7d, 1d, -2d, -4d, -1d, 31d, -1d, -3d};

        int[] sparseArray2 = {2, 6, 1};
        double[] sparseArray2Values = {1d, 11d, 2d};

        int[] sparseArray3 = {4, 6, 1};
        double[] sparseArray3Values = {52d, 71d, 11d};

        int[] sparseArray4 = {4, 1, 2};
        double[] sparseArray4Values = {17d, 7d, 9d};

        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create(3d, 4d, new SparseVector(20, sparseArray1, sparseArray1Values)),
                RowFactory.create(4d, 5d, new SparseVector(20, sparseArray2, sparseArray2Values)),
                RowFactory.create(5d, 5d, new SparseVector(20, sparseArray3, sparseArray3Values)),
                RowFactory.create(6d, 5d, new SparseVector(20, sparseArray4, sparseArray4Values))
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("value1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("vector1", new VectorUDT(), false, Metadata.empty())
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);
        VectorBinarizer vectorBinarizer = new VectorBinarizer()
                .setInputCol("vector1")
                .setOutputCol("binarized");


        //Export this model
        byte[] exportedModel = ModelExporter.export(vectorBinarizer, null);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        //compare predictions
        Row[] sparkOutput = vectorBinarizer.transform(df).orderBy("id").select("id", "value1", "vector1", "binarized").collect();
        for (Row row : sparkOutput) {

            Map<String, Object> data = new HashMap<>();
            data.put(vectorBinarizer.getInputCol(), ((SparseVector) row.get(2)).toArray());
            transformer.transform(data);
            double[] output = (double[]) data.get(vectorBinarizer.getOutputCol());
            assertArrayEquals(output, ((SparseVector)row.get(3)).toArray(), 0d);
        }
    }
}
