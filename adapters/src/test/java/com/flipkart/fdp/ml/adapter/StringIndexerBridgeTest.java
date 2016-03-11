package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.StringIndexerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.types.DataTypes.*;
import static org.junit.Assert.assertEquals;

/**
 * Created by akshay.us on 3/2/16.
 */
public class StringIndexerBridgeTest extends SparkTestBase {

    @Test
    public void testStringIndexer() {

        //prepare data
        StructType schema = createStructType(new StructField[]{
                createStructField("id", IntegerType, false),
                createStructField("label", StringType, false)
        });
        List<Row> data = Arrays.asList(
                cr(0, "a"), cr(1, "b"), cr(2, "c"), cr(3, "a"), cr(4, "a"), cr(5, "c"));
        DataFrame dataset = sqlContext.createDataFrame(data, schema);

        //train model in spark
        StringIndexerModel model = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("labelIndex").fit(dataset);

        //Export this model
        byte[] exportedModel = ModelExporter.export(model, dataset);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        //TODO: fix the transformer interface for this use case
        StringIndexerTransformer stringIndexerTransformer = (StringIndexerTransformer) transformer;

        //compare predictions
        Row[] sparkOutput = model.transform(dataset).orderBy("id").select("id", "label", "labelIndex").collect();
        for (Row row : sparkOutput) {
            double indexerOutput = (double)stringIndexerTransformer.transform(new String []{ (String)row.get(1) });
            assertEquals(indexerOutput, (double) row.get(2), 0.01);
        }

    }

    /**
     * An alias for RowFactory.create.
     */
    private Row cr(Object... values) {
        return RowFactory.create(values);
    }
}
