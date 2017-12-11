package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.StringMerge;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class StringMergeBridgeTest extends SparkTestBase {
	@Test
	public void testStringMerge() {

		//prepare data
		JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, "string1", "string2"),
				RowFactory.create(1, "first part of string", "second part of string")
		));

		StructType schema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("input1", DataTypes.StringType, true, Metadata.empty()),
				new StructField("input2", DataTypes.StringType, true, Metadata.empty())
		});
		Dataset<Row> df = spark.createDataFrame(rdd, schema);

		//train model in spark
		StringMerge sparkModel = new StringMerge()
				.setInputCol1("input1")
				.setInputCol2("input2")
				.setOutputCol("output");
		//Export this model
		byte[] exportedModel = ModelExporter.export(sparkModel);

//		//Import and get Transformer
		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
//
//		//compare predictions
		List<Row> sparkOutput = sparkModel.transform(df).orderBy("id").select("input1", "input2", "output").collectAsList();
		for (Row row : sparkOutput) {

			Map<String, Object> data = new HashMap<String, Object>();
			data.put(sparkModel.getInputCol1(), row.get(0));
			data.put(sparkModel.getInputCol2(), row.get(1));
			transformer.transform(data);
			String actual = (String) data.get(sparkModel.getOutputCol());

			assertEquals(actual, row.get(2));
		}
	}
}
