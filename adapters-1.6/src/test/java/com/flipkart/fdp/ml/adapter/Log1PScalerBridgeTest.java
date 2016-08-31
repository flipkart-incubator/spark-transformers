package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.Log1PScaler;
import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by akshay.us on 3/14/16.
 */
public class Log1PScalerBridgeTest extends SparkTestBase {

    private final double data[][] = {{10.0, 2.3, 0.6},
            {0.1, 5.1, 1.0},
            {1.7, 33.6, 3.3}};


    @Test
    public void testCustomScalerDenseVector() {
        final double precomputedAns[][] = new double[3][3];
        //precompute answers
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    precomputedAns[j][k] = Math.log1p(data[j][k]);

        //prepare data
        List<LabeledPoint> localTraining = Arrays.asList(
                new LabeledPoint(1.0, Vectors.dense(data[0])),
                new LabeledPoint(2.0, Vectors.dense(data[1])),
                new LabeledPoint(3.0, Vectors.dense(data[2])));
        DataFrame df = sqlContext.createDataFrame(sc.parallelize(localTraining), LabeledPoint.class);

        for (int i = 0; i < 2; i++) {
            //train model in spark
            Log1PScaler sparkModel = new Log1PScaler()
                    .setInputCol("features")
                    .setOutputCol("scaledOutput");

            //Export model, import it back and get transformer
            byte[] exportedModel = ModelExporter.export(sparkModel, df);
            final Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

            //compare predictions
            Row[] sparkOutput = sparkModel.transform(df).orderBy("label").select("features", "scaledOutput").collect();
            assertCorrectness(sparkOutput, precomputedAns, transformer);
        }
    }

    @Test
    public void testCustomScalerSparseVector() {
        //prepare data
        List<LabeledPoint> localTraining = Arrays.asList(
                new LabeledPoint(1.0, Vectors.sparse(5, new int[]{0, 1, 2}, data[0])),
                new LabeledPoint(2.0, Vectors.sparse(5, new int[]{1, 2, 3}, data[1])),
                new LabeledPoint(3.0, Vectors.sparse(5, new int[]{2, 3, 4}, data[2])));

        DataFrame df = sqlContext.createDataFrame(sc.parallelize(localTraining), LabeledPoint.class);

        //train model in spark
        Log1PScaler sparkModel = new Log1PScaler()
                .setInputCol("features")
                .setOutputCol("scaledOutput");

        //Export model, import it back and get transformer
        byte[] exportedModel = ModelExporter.export(sparkModel, df);
        final Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        Row[] sparkOutput = sparkModel.transform(df).orderBy("label").select("features", "scaledOutput").collect();

        for (int i = 0; i < 2; i++) {
            double[] input = ((Vector) sparkOutput[i].get(0)).toArray();

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("features", input);
            transformer.transform(data);
            double[] transformedOp = (double[]) data.get("scaledOutput");

            double[] sparkOp = ((Vector) sparkOutput[i].get(1)).toArray();
            assertArrayEquals(transformedOp, sparkOp, EPSILON);
        }
    }


    private void assertCorrectness(Row[] sparkOutput, double[][] expected, Transformer transformer) {
        for (int i = 0; i < 2; i++) {
            double[] input = ((Vector) sparkOutput[i].get(0)).toArray();

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("features", input);
            transformer.transform(data);
            double[] transformedOp = (double[]) data.get("scaledOutput");

            double[] sparkOp = ((Vector) sparkOutput[i].get(1)).toArray();
            assertArrayEquals(transformedOp, sparkOp, EPSILON);
            assertArrayEquals(transformedOp, expected[i], EPSILON);
        }
    }
}
