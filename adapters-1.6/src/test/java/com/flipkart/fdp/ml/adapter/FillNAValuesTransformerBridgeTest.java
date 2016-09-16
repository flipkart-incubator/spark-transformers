package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.FillNAValuesTransformer;
import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Double.NaN;
import static org.apache.spark.sql.types.DataTypes.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by akshay.us on 9/15/16.
 */
public class FillNAValuesTransformerBridgeTest extends SparkTestBase {

    @Test
    public void shouldBehaveExactlyAsSparkNAFillerForAllSupportedDataTypes() {

        DataFrame df = getDataFrame();
        DataFrame df1 = df.na().fill( getFillNAMap() );

        FillNAValuesTransformer fillNAValuesTransformer = new FillNAValuesTransformer();
        fillNAValuesTransformer.setNAValueMap( getFillNAMap() );
        DataFrame df2 = fillNAValuesTransformer.transform(df);

        Row[] data1 = df1.orderBy("id").select("id", "a", "b", "c", "d").collect();
        Row[] data2 = df2.orderBy("id").select("id", "a", "b", "c", "d").collect();

        for( int i =0; i < data1.length; i++) {
            for( int j=1; j<=4; j++) {
                assertEquals(data1[i].get(j), data2[i].get(j));
            }
        }
    }

    @Test
    public void shouldBehaveExactlySameAfterExportForAllSupportedDataTypes() {

        DataFrame df = getDataFrame();
        Row[] originalData = df.orderBy("id").select("id", "a", "b", "c", "d").collect();
        FillNAValuesTransformer fillNAValuesTransformer = new FillNAValuesTransformer();
        fillNAValuesTransformer.setNAValueMap( getFillNAMap() );
        Row[] sparkOutput = fillNAValuesTransformer.transform(df).orderBy("id").select("id", "a", "b", "c", "d").collect();

        byte[] exportedModel = ModelExporter.export(fillNAValuesTransformer, df);
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        for( int i=0; i < originalData.length; i++) {
            Map<String, Object> input = new HashMap<String, Object>();
            input.put("a", originalData[i].get(1));
            input.put("b", originalData[i].get(2));
            input.put("c", originalData[i].get(3));
            input.put("d", originalData[i].get(4));

            transformer.transform(input);

            assertEquals(sparkOutput[i].get(1), input.get("a"));
            assertEquals(sparkOutput[i].get(2), input.get("b"));
            assertEquals(sparkOutput[i].get(3), input.get("c"));
            assertEquals(sparkOutput[i].get(4), input.get("d"));
        }
    }

    @Test
    public void shouldWorkCorrectlyWithPipeline() {

        //Prepare test data
        DataFrame df = getDataFrame();
        Row[] originalData = df.orderBy("id").select("id", "a", "b", "c", "d").collect();

        //prepare transformation pipeline
        FillNAValuesTransformer fillNAValuesTransformer = new FillNAValuesTransformer();
        fillNAValuesTransformer.setNAValueMap( getFillNAMap() );
        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{fillNAValuesTransformer});
        PipelineModel model = pipeline.fit(df);

        //predict
        Row[] sparkOutput = model.transform(df).orderBy("id").select("id", "a", "b", "c", "d").collect();

        //export
        byte[] exportedModel = ModelExporter.export(model, df);
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //verify correctness
        assertTrue(transformer.getInputKeys().size() == 4);
        assertTrue(transformer.getInputKeys().containsAll(Arrays.asList("a", "b", "c", "d")));
        assertTrue(transformer.getOutputKeys().size() == 4);
        assertTrue(transformer.getOutputKeys().containsAll(Arrays.asList("a", "b", "c", "d")));
        for( int i=0; i < originalData.length; i++) {
            Map<String, Object> input = new HashMap<String, Object>();
            input.put("a", originalData[i].get(1));
            input.put("b", originalData[i].get(2));
            input.put("c", originalData[i].get(3));
            input.put("d", originalData[i].get(4));

            transformer.transform(input);

            assertEquals(sparkOutput[i].get(1), input.get("a"));
            assertEquals(sparkOutput[i].get(2), input.get("b"));
            assertEquals(sparkOutput[i].get(3), input.get("c"));
            assertEquals(sparkOutput[i].get(4), input.get("d"));
        }
    }

    private Map<String, Object> getFillNAMap() {
        Map<String, Object> map = new HashMap();
        map.put("a", "testString");
        map.put("b", 3.14);
        map.put("c", 42);
        map.put("d", false);
        return map;
    }

    private DataFrame getDataFrame() {

        StructType schema = createStructType(new StructField[]{
                createStructField("id", IntegerType, false),
                createStructField("a", StringType, false),
                createStructField("b", DoubleType, false),
                createStructField("c", DoubleType, false),
                createStructField("d", BooleanType, false),

        });
        List<Row> trainingData = Arrays.asList(
                cr(1, null, null, null, null),
                cr(2, "test", 1.2, null, null),
                cr(3, null, 1.1, null, false),
                cr(4, "faffa", NaN, 45.0, true)
        );

        DataFrame df = sqlContext.createDataFrame(trainingData, schema);
        return df;
    }
}
