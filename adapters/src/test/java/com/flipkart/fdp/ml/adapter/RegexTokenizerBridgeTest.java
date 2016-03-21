package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.spark.sql.types.DataTypes.*;

/**
 * Created by akshay.us on 3/14/16.
 */
public class RegexTokenizerBridgeTest extends SparkTestBase {

    @Test
    public void testRegexTokenizer() {

        //prepare data
        StructType schema = createStructType(new StructField[]{
                createStructField("rawText", StringType, false),
        });
        List<Row> trainingData = Arrays.asList(
                cr("Test of tok."),
                cr("Te,st.  punct")
        );
        DataFrame dataset = sqlContext.createDataFrame(trainingData, schema);

        //train model in spark
        RegexTokenizer sparkModel = new RegexTokenizer()
                .setInputCol("rawText")
                .setOutputCol("tokens")
                .setPattern("\\s")
                .setGaps(true)
                .setToLowercase(false)
                .setMinTokenLength(3);

        //Export this model
        byte[] exportedModel = ModelExporter.export(sparkModel, dataset);

        //Import and get Transformer
        Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

        Row[] pairs = sparkModel.transform(dataset).select("rawText", "tokens").collect();
        for (Row row : pairs) {

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("input", (String) row.get(0));
            transformer.transform(data);
            String[] output = (String[]) data.get("output");

            Object sparkOp = row.get(1);
            System.out.println(ArrayUtils.toString(output));
            System.out.println(row.get(1));
        }
    }

}
