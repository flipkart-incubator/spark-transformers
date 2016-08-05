package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.CustomOneHotEncoder;
import com.flipkart.fdp.ml.CustomOneHotEncoderModel;
import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.linalg.Vector;
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
import static org.junit.Assert.assertEquals;

/**
 * Created by shubhranshu.shekhar on 21/06/16.
 */
public class CustomOneHotEncoderBridgeTest  extends SparkTestBase {
    @Test
    public void testCustomOneHotEncoding() {
        // prepare data
        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create(0d, "a"),
                RowFactory.create(1d, "b"),
                RowFactory.create(2d, "c"),
                RowFactory.create(3d, "a"),
                RowFactory.create(4d, "a"),
                RowFactory.create(5d, "c")
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("category", DataTypes.StringType, false, Metadata.empty())
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);
        StringIndexerModel indexer = new StringIndexer()
                .setInputCol("category")
                .setOutputCol("categoryIndex")
                .fit(df);
        DataFrame indexed = indexer.transform(df);

        CustomOneHotEncoderModel sparkModel = new CustomOneHotEncoder()
                .setInputCol("categoryIndex")
                .setOutputCol("categoryVec")
                .fit(indexed);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel, indexed);

        //Create spark's OneHotEncoder
        OneHotEncoder sparkOneHotModel = new OneHotEncoder()
                .setInputCol("categoryIndex")
                .setOutputCol("categoryVec");

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        //compare predictions
        Row[] sparkOutput = sparkModel.transform(indexed).orderBy("id").select("id", "categoryIndex", "categoryVec").collect();
        Row[] sparkOneHotOutput = sparkOneHotModel.transform(indexed).orderBy("id").select("id", "categoryIndex", "categoryVec").collect();

        //Compare Spark's OneHotEncoder with CustomOneHotEncoder
        //See if the dictionary size is equal
        assertEquals(sparkOutput.length, sparkOneHotOutput.length);

        for (int i = 0; i < sparkOutput.length; i++) {
            Row row= sparkOutput[i];
            Map<String, Object> data = new HashMap<String, Object>();
            data.put(sparkModel.getInputCol(), row.getDouble(1));
            transformer.transform(data);
            double[] transformedOp = (double[]) data.get(sparkModel.getOutputCol());

            double[] sparkOp = ((Vector) row.get(2)).toArray();
            //get spark's OneHotEncoder output
            double[] sparkOneHotOp = ((Vector) sparkOneHotOutput[i].get(2)).toArray();
            assertArrayEquals(transformedOp, sparkOp, 0.01);
            assertArrayEquals(sparkOneHotOp, sparkOp, 0.01);
        }
    }
}
