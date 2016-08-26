package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.ProbabilityTransform;
import com.flipkart.fdp.ml.ProbabilityTransformModel;
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
public class ProbabilityTransformBridgeTest extends SparkTestBase {
    private final double data[] = {0.0, 0.5, 0.98};

    //private final
    private final double p1 = 1.0/61;
    private final double r1 = 1.0/4;
    private final int idx = 1;

    private double[] getTrueProb(double[] prob, double p1, double r1){
        double[] trueProb = new double[3];
        for(int i=0; i < prob.length; i++){
                trueProb[i] = (prob[i] *p1/r1) / ((prob[i] *p1/r1) + ((1-prob[i]) *(1-p1)/(1-r1)));
        }
        return trueProb;
    }

    @Test
    public void testProbabilityTransform(){
        //get expected true probability
        double[] trueProb = getTrueProb(data, this.p1, this.r1);
        // prepare data
        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create((data[0])),
                RowFactory.create((data[1])),
                RowFactory.create((data[2]))
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("probability", DataTypes.DoubleType, false, Metadata.empty())
                //new StructField("probability", new VectorUDT(), false, Metadata.empty()),
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);

        ProbabilityTransformModel customSparkModel = new ProbabilityTransform()
                .setInputCol("probability")
                .setOutputCol("trueProbability")
                .setActualClickProportion(p1)
                .setUnderSampledClickProportion(r1)
                .setProbIndex(idx)
                .fit(df);

        //Export this model
        byte[] exportedModel = ModelExporter.export(customSparkModel, df);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        Row[] customSparkOutput = customSparkModel.transform(df).select("probability", "trueProbability").collect();

        for (int i = 0; i < customSparkOutput.length; i++) {
            Row row= customSparkOutput[i];
            Map<String, Object> mapData = new HashMap<String, Object>();
            mapData.put(transformer.getInputKeys().iterator().next(), row.getDouble(0));
            transformer.transform(mapData);
            double transformedOp = (double) mapData.get(transformer.getOutputKeys().iterator().next());

            double sparkOp = ((double) row.getDouble(1));
            //Check if imported model produces same result as spark output
            assertEquals(transformedOp, sparkOp, 0.000001);
            //check if spark output is correct. This also tests for correctness of ProbabilityTransform
            assertEquals(trueProb[i], sparkOp, 0.000001);

        }
    }
}
