package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.ChiSqSelectorModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
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


/**
 * Created by rohan.shetty on 28/03/16.
 */

public class ChiSqSelectorBridgeTest extends SparkTestBase {

    @Test
    public void testChiSqSelector() {
        // prepare data

        List<Row> inputData = Arrays.asList(
                RowFactory.create(0d, 0d, new DenseVector(new double[] {8d,7d,0d})),
                RowFactory.create(1d, 1d, new DenseVector(new double[] {0d,9d, 6d})),
                RowFactory.create(2d, 1d, new DenseVector(new double[] {0.0d, 9.0d, 8.0d})),
                RowFactory.create(3d, 2d, new DenseVector(new double[] {8.0d, 9.0d, 5.0d}))
        );

        double[] preFilteredData = { 0.0d, 6.0d, 8.0d, 5.0d };

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(inputData, schema);
        ChiSqSelector chiSqSelector = new ChiSqSelector();
        chiSqSelector.setNumTopFeatures(1);
        chiSqSelector.setFeaturesCol("features");
        chiSqSelector.setLabelCol("label");
        chiSqSelector.setOutputCol("output");

        ChiSqSelectorModel chiSqSelectorModel = chiSqSelector.fit(df);

        //Export this model
        byte[] exportedModel = ModelExporter.export(chiSqSelectorModel);

        String exportedModelJson = new String (exportedModel);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        List<Row> sparkOutput = chiSqSelectorModel.transform(df).orderBy("id").select("id", "label", "features","output").collectAsList();
        for (Row row : sparkOutput) {
            Map<String, Object> data = new HashMap<>();
            data.put(chiSqSelectorModel.getFeaturesCol(), ((DenseVector) row.get(2)).toArray());
            transformer.transform(data);
            double[] output = (double[]) data.get(chiSqSelectorModel.getOutputCol());
            System.out.println(Arrays.toString(output));
            assertArrayEquals(output,((DenseVector) row.get(3)).toArray(), 0d);
        }
    }
}

