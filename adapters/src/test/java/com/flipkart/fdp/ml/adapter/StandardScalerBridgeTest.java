package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
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
public class StandardScalerBridgeTest extends SparkTestBase {

    private final double data[][] = {{-2.0, 2.3, 0.0},
            {0.0, -5.1, 1.0},
            {1.7, -0.6, 3.3}};

    private final double resWithMean[][] = {{-1.9, 3.433333333333, -1.433333333333},
            {0.1, -3.966666666667, -0.433333333333},
            {1.8, 0.533333333333, 1.866666666667}};

    private final double resWithStd[][] = {{-1.079898494312, 0.616834091415, 0.0},
            {0.0, -1.367762550529, 0.590968109266},
            {0.917913720165, -0.160913241239, 1.950194760579}};

    private final double resWithBoth[][] = {{-1.0259035695965, 0.920781324866, -0.8470542899497},
            {0.0539949247156, -1.063815317078, -0.256086180682},
            {0.9719086448809, 0.143033992212, 1.103140470631}};

    @Test
    public void testStandardScaler() {
        //prepare data
        List<LabeledPoint> localTraining = Arrays.asList(
                new LabeledPoint(1.0, Vectors.dense(data[0])),
                new LabeledPoint(2.0, Vectors.dense(data[1])),
                new LabeledPoint(3.0, Vectors.dense(data[2])));
        DataFrame df = sqlContext.createDataFrame(sc.parallelize(localTraining), LabeledPoint.class);



        //train model in spark
        StandardScalerModel sparkModelNone = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledOutputNone")
                .setWithMean(false)
                .setWithStd(false)
                .fit(df);

        StandardScalerModel sparkModelWithMean = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledOutputWithMean")
                .setWithMean(true)
                .setWithStd(false)
                .fit(df);

        StandardScalerModel sparkModelWithStd = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledOutputWithStd")
                .setWithMean(false)
                .setWithStd(true)
                .fit(df);

        StandardScalerModel sparkModelWithBoth = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledOutputWithBoth")
                .setWithMean(true)
                .setWithStd(true)
                .fit(df);


        //Export model, import it back and get transformer
        byte[] exportedModel = ModelExporter.export(sparkModelNone, df);
        final Transformer transformerNone = ModelImporter.importAndGetTransformer(exportedModel);

        exportedModel = ModelExporter.export(sparkModelWithMean, df);
        final Transformer transformerWithMean = ModelImporter.importAndGetTransformer(exportedModel);

        exportedModel = ModelExporter.export(sparkModelWithStd, df);
        final Transformer transformerWithStd = ModelImporter.importAndGetTransformer(exportedModel);

        exportedModel = ModelExporter.export(sparkModelWithBoth, df);
        final Transformer transformerWithBoth = ModelImporter.importAndGetTransformer(exportedModel);


        //compare predictions
        Row[] sparkNoneOutput = sparkModelNone.transform(df).orderBy("label").select("features", "scaledOutputNone").collect();
        assertCorrectness(sparkNoneOutput, data, transformerNone);

        Row[] sparkWithMeanOutput = sparkModelWithMean.transform(df).orderBy("label").select("features", "scaledOutputWithMean").collect();
        assertCorrectness(sparkWithMeanOutput, resWithMean, transformerWithMean);

        Row[] sparkWithStdOutput = sparkModelWithStd.transform(df).orderBy("label").select("features", "scaledOutputWithStd").collect();
        assertCorrectness(sparkWithStdOutput, resWithStd, transformerWithStd);

        Row[] sparkWithBothOutput = sparkModelWithBoth.transform(df).orderBy("label").select("features", "scaledOutputWithBoth").collect();
        assertCorrectness(sparkWithBothOutput, resWithBoth, transformerWithBoth);

    }

    private void assertCorrectness(Row[] sparkOutput, double[][] expected, Transformer transformer) {
        for( int i = 0 ; i < 2; i++) {
            double[] input = ((Vector) sparkOutput[i].get(0)).toArray();

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("input",input);
            transformer.transform(data);
            double[] transformedOp = (double []) data.get("output");

            double[] sparkOp = ((Vector) sparkOutput[i].get(1)).toArray();
            assertArrayEquals(transformedOp, sparkOp, 0.01);
            assertArrayEquals(transformedOp, expected[i], 0.01);
        }
    }
}
