package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.StringSanitizer;
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

import static org.junit.Assert.assertTrue;

public class StringSanitizerBridgeTest extends SparkTestBase {
	@Test
	public void testStringSanitizer() {

		//prepare data
		JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, "Jyoti complex near Sananda clothes store; English Bazar; Malda;WB;India,"),
				RowFactory.create(2, "hallalli vinayaka tent road c/o B K vishwanath Mandya"),
				RowFactory.create(3, "M.sathish S/o devudu Lakshmi opticals Gokavaram bus stand Rajhamundry 9494954476")
		));

		StructType schema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("rawText", DataTypes.StringType, false, Metadata.empty())
		});
		Dataset<Row> dataset = spark.createDataFrame(rdd, schema);
		dataset.show();

		//train model in spark
		StringSanitizer sparkModel = new StringSanitizer()
				.setInputCol("rawText")
				.setOutputCol("token");

		//Export this model
		byte[] exportedModel = ModelExporter.export(sparkModel);

		//Import and get Transformer
		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

		List<Row> pairs = sparkModel.transform(dataset).select("rawText", "token").collectAsList();

		for (Row row : pairs) {
			Map<String, Object> data = new HashMap<String, Object>();
			data.put(sparkModel.getInputCol(), row.getString(0));
			transformer.transform(data);

			String[] actual = (String[]) data.get(sparkModel.getOutputCol());

			List<String> actualList = Arrays.asList(actual);
			List<String> expected = row.getList(1);

			assertTrue("both should be same", actualList.equals(expected));
		}
	}
}
