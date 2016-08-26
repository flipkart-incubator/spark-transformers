package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.AlgebraicTransform;
import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
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

import static org.junit.Assert.assertEquals;

/**
 * Created by shubhranshu.shekhar on 18/08/16.
 */
public class AlgebraicTransformBridgeTest extends SparkTestBase {
    private final double data[] = {0.0, 0.5, 0.98};

    private final double[] coeff = {5, 8};//{b, a} for ax + b

    private double[] axBTranform(double[] coeff, double[] dataPoint){
        double[] retArr = new double[dataPoint.length];
        for (int j = 0; j < dataPoint.length; j++){
            if(coeff.length == 0){
                retArr[j] = 0.0;
            }
            else{
                double sum = coeff[0];
                for(int i = 1; i < coeff.length; i++){
                    sum = sum + coeff[i] * Math.pow(dataPoint[j], i);
                }
                retArr[j] = sum;
            }
        }
        return retArr;
    }

    @Test
    public void testAlgebraicTransform(){
        //get expected Ax + b transform for given data
        double[] axB = axBTranform(this.coeff, this.data);
        // prepare data
        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create((data[0])),
                RowFactory.create((data[1])),
                RowFactory.create((data[2]))
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("trueProb", DataTypes.DoubleType, false, Metadata.empty())
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);

        AlgebraicTransform customSparkModel = new AlgebraicTransform()
                .setInputCol("trueProb")
                .setOutputCol("scaledProb")
                .setCoefficients(coeff);

        //Export this model
        byte[] exportedModel = ModelExporter.export(customSparkModel, df);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        Row[] customSparkOutput = customSparkModel.transform(df).select("trueProb", "scaledProb").collect();

        for (int i = 0; i < customSparkOutput.length; i++) {
            Row row= customSparkOutput[i];
            Map<String, Object> mapData = new HashMap<String, Object>();
            mapData.put(customSparkModel.getInputCol(), row.getDouble(0));
            transformer.transform(mapData);
            double transformedOp = (double) mapData.get(customSparkModel.getOutputCol());

            double sparkOp = ((double) row.getDouble(1));
            //Check if imported model produces same result as spark output
            assertEquals(transformedOp, sparkOp, 0.000001);
            //check if spark output is correct. This also tests for correctness of AlgebraicTransform
            assertEquals(axB[i], sparkOp, 0.000001);

        }
    }
}
