package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.*;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Assert;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AISPipelineBridgeTest extends SparkTestBase {
	@Test
	public void testPipeline() {
		//prepare data
		JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, "Kakinada, Andhra Pradesh, gandhinagar.near:kkr's gowtham model school .venkateswaraswamy temple","venkateswaraswmy temple", 0.0),
				RowFactory.create(2, "Lake Garden appt, E2/24,3rd floor ,6th main road,Mogappair Eri Scheme,Chennai-37","SBIOA", 0.0),
				RowFactory.create(3, "71/2RT SAIDABAD COLONY","Ramalayam Temple Arch.", 0.0),
				RowFactory.create(4, "To badalpra , ta veraval, dis gir somnatha,vaya prbhaspatan, post kajli ,cite veraval",",VERAVAL,IN,Gir somnatha,362268", 1.0)
		));

		StructType schema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("addressLine1", DataTypes.StringType, true, Metadata.empty()),
				new StructField("addressLine2", DataTypes.StringType, true, Metadata.empty()),
				new StructField("label", DataTypes.DoubleType, false, Metadata.empty())
		});
		Dataset<Row> trainingData = spark.createDataFrame(rdd, schema);
		trainingData.show(false);

		//train model in spark
		StringMerge stringMerge = new StringMerge()
				.setInputCol1("addressLine1")
				.setInputCol2("addressLine2")
				.setOutputCol("mergedAddress");

		StringSanitizer stringSanitizer = new StringSanitizer()
				.setInputCol(stringMerge.getOutputCol())
				.setOutputCol("sanitizedAddress");

		CommonAddressFeatures commonAddressFeatures = new CommonAddressFeatures()
				.setInputCol(stringSanitizer.getOutputCol())
				.setRawInputCol(stringMerge.getOutputCol());

		PopularWordsEstimator popularWordsEstimator = new PopularWordsEstimator()
				.setInputCol("sanitizedAddress")
				.setOutputCol("commonFraction");

		String[] featureColumns = new String[]{"commonFraction", "numWords", "numCommas", "numericPresent", "addressLength", "favouredStart", "unfavouredStart"};

		VectorAssembler vectorAssembler = new VectorAssembler()
				.setOutputCol("features")
				.setInputCols(featureColumns);

		LogisticRegression logisticRegression = new LogisticRegression()
				.setLabelCol("label")
				.setFeaturesCol("features");

		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[]{stringMerge, stringSanitizer, commonAddressFeatures, popularWordsEstimator, vectorAssembler, logisticRegression});

		PipelineModel pipelineModel = pipeline.fit(trainingData);

		//Export this model
		byte[] exportedModel = ModelExporter.export(pipelineModel);

		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

//		//test
		JavaRDD<Row> testRdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(3, "VADAKKKIYAL HOUSE VELIMUKKU SOUTH PO","CALICUT UNIVERSITY"),
				RowFactory.create(3, "Papu ate ki chakkki nagla mallah dodhpur civil line aligarh",""),
				RowFactory.create(3, "hallalli vinayaka tent road c/o B K vishwanath Mandya","harishchandra circle"),
				RowFactory.create(3, "","")
		));
//
		StructType testSchema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("addressLine1", DataTypes.StringType, true, Metadata.empty()),
				new StructField("addressLine2", DataTypes.StringType, true, Metadata.empty())
		});
//
		Dataset<Row> testData = spark.createDataFrame(testRdd, testSchema);
		Dataset<Row> transform = pipelineModel.transform(testData);
		transform.show(false);

		Dataset<Row> rowDataset = transform.select("id", "addressLine1", "addressLine2", "mergedAddress", "sanitizedAddress", "numWords", "numCommas", "numericPresent", "addressLength", "favouredStart", "unfavouredStart", "commonFraction", "features", "rawPrediction", "probability", "prediction");
		rowDataset.show(false);


		for (Row row : rowDataset.collectAsList()) {
			Map<String, Object> data = null;
			long start = System.currentTimeMillis();
			int count = 100000;
			for (int i = 0; i < count; i++) {
				data = new HashMap<String, Object>();
				data.put("addressLine1", row.getString(1));
				data.put("addressLine2", row.getString(2));
				transformer.transform(data);
			}
			long end = System.currentTimeMillis();
			System.out.println("Time taken = " + (end - start) / ((double) count));

			Assert.assertEquals("output should be same", row.getString(3), (String) data.get("mergedAddress"));

			List<String> actualSanitizedAddress = Arrays.asList((String[]) data.get("sanitizedAddress"));

			List<String> expectedSanitizedAddress = row.getList(4);

			assertTrue("both should be same", expectedSanitizedAddress.equals(actualSanitizedAddress));

			assertEquals("number of words should be equals", row.get(5), data.get("numWords"));
			assertEquals("number of commas should be equals", row.get(6), data.get("numCommas"));
			assertEquals("numericPresent should be equals", row.get(7), data.get("numericPresent"));
			assertEquals("addressLength should be equals", row.get(8), data.get("addressLength"));
			assertEquals("favouredStart should be equals", row.get(9), data.get("favouredStart"));
			assertEquals("unfavouredStart should be equals", row.get(10), data.get("unfavouredStart"));

			double expectedCommonFraction = row.getDouble(11);
			double actualCommonFraction = (double) data.get("commonFraction");
			assertEquals(expectedCommonFraction, actualCommonFraction, 0.01);

			DenseVector denseVector = (DenseVector) row.get(13);
			double expectedRawPred = denseVector.toArray()[0];
			double expectedRawPrediction = 1.0 / (1.0 + Math.exp(-expectedRawPred));
			double actualRawPrediction = 1 - (double) data.get("probability");

			assertEquals(expectedRawPrediction, actualRawPrediction, 0.0000001);
		}
	}
}
