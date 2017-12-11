package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.PopularWordsEstimator;
import org.apache.spark.ml.PopularWordsModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class PopularWordsEstimatorBridgeTest extends SparkTestBase {
	@Test
	public void testPopularWordsEstimator() {
		final String[] addressLine1 = new String[]{
				"Jyoti complex near Sananda clothes store; English Bazar; Malda;WB;India",
				"hallalli vinayaka tent road c/o B K vishwanath Mandya",
				"M.sathish S/o devudu Lakshmi opticals Gokavaram bus stand Rajhamundry 9494954476"
		};

		final String[] addressLine2 = new String[]{
				"",
				"harishchandra circle",
				"Near Lilly's Textile"
		};
		final String[] mergeAddress = new String[]{
				addressLine1[0] + " " + addressLine2[0],
				addressLine1[1] + " " + addressLine2[1],
				addressLine1[2] + " " + addressLine2[2]
		};

		final List<String[]> sanitizedAddress = new ArrayList<>();
		sanitizedAddress.add(mergeAddress[0].split(" "));
		sanitizedAddress.add(mergeAddress[1].split(" "));
		sanitizedAddress.add(mergeAddress[2].split(" "));
		//prepare data
		JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, sanitizedAddress.get(0), 0.0),
				RowFactory.create(1, sanitizedAddress.get(1), 0.0),
				RowFactory.create(1, sanitizedAddress.get(2), 1.0)
		));

		StructType schema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("sanitizedAddress", new ArrayType(DataTypes.StringType, true), false, Metadata.empty()),
				new StructField("label", DataTypes.DoubleType, false, Metadata.empty())

		});
		Dataset<Row> dataset = spark.createDataFrame(rdd, schema);
		dataset.show(false);

		PopularWordsModel sparkModel = new PopularWordsEstimator()
				.setInputCol("sanitizedAddress")
				.setOutputCol("commonFraction")

				.fit(dataset);

		byte[] exportedModel = ModelExporter.export(sparkModel);

		//Import and get Transformer
		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

		Dataset<Row> rowDataset = sparkModel.transform(dataset).orderBy("id").select("sanitizedAddress", "commonFraction");
		rowDataset.show(false);

		assertCorrectness(rowDataset, transformer);
	}

	private void assertCorrectness(Dataset<Row> rowDataset, Transformer transformer) {
		List<Row> sparkOutput = rowDataset.collectAsList();
		for (Row row : sparkOutput) {
			List<Object> list = row.getList(0);
			String[] sanitizedAddress = new String[list.size()];
			for (int j = 0; j < sanitizedAddress.length; j++) {
				sanitizedAddress[j] = (String) list.get(j);
			}

			Map<String, Object> data = new HashMap<>();
			data.put("sanitizedAddress", sanitizedAddress);

			double expected = row.getDouble(1);
			transformer.transform(data);
			double actual = (double) data.get("commonFraction");

			assertEquals(expected, actual, 0.01);
		}
	}
}
