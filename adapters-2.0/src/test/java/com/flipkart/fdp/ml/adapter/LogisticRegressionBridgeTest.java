package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import com.google.common.collect.Lists;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;


import java.util.Arrays;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.types.DataTypes.*;

public class LogisticRegressionBridgeTest extends SparkTestBase {

	@Test
	public void testLogisticRegression() {
		//prepare data
		String datapath = "/Users/gaurav.prasad/gitCurrent/github/my/spark-transformers/adapters-2.0/src/test/resources/binary_classification_test.libsvm";
		JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

		//Train model in spark
		LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(trainingData.rdd());

		//Export this model
		byte[] exportedModel = ModelExporter.export(lrmodel);

		//Import and get Transformer
		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

		//validate predictions
		List<LabeledPoint> testPoints = trainingData.collect();
		for (LabeledPoint i : testPoints) {
			Vector v = i.features();
			double actual = lrmodel.predict(v);

			Map<String, Object> data = new HashMap<String, Object>();
			data.put("features", v.toArray());
			transformer.transform(data);
			double predicted = (double) data.get("prediction");

			assertEquals(actual, predicted, 0.01);
		}
	}

//	@Test
	public void testLogisticRegression1() {
		List<Row> inputData = Arrays.asList(
				RowFactory.create(0, new DenseVector(new double[]{8d, 7d, 0d}), 0.0),
				RowFactory.create(1, new DenseVector(new double[]{0d, 9d, 6d}), 0.0),
				RowFactory.create(1, new DenseVector(new double[]{1d, 3d, 6d}), 0.0),
				RowFactory.create(1, new DenseVector(new double[]{2d, 4d, 7d}), 1.0),
				RowFactory.create(1, new DenseVector(new double[]{3d, 5d, 8d}), 0.0),
				RowFactory.create(2, new DenseVector(new double[]{0.0d, 9.0d, 8.0d}), 1.0),
				RowFactory.create(3, new DenseVector(new double[]{8.0d, 9.9d, 5.0d}), 1.0)
		);

		StructType schema = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("features", new VectorUDT(), false, Metadata.empty()),
				new StructField("label", DataTypes.DoubleType, false, Metadata.empty())
		});

		Dataset<Row> trainingData = spark.createDataFrame(inputData, schema);
		trainingData.show(false);
//
		org.apache.spark.ml.classification.LogisticRegressionModel sparkModel = new LogisticRegression()
				.setLabelCol("label")
				.setFeaturesCol("features")
				.fit(trainingData);

		//Export this model
		byte[] exportedModel = ModelExporter.export(sparkModel);

		//Import and get Transformer
		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

		System.out.println(sparkModel.numFeatures());
		System.out.println(sparkModel.numClasses());
		System.out.println(sparkModel.intercept());
		System.out.println(sparkModel.coefficients());
		System.out.println("done");


		JavaRDD<Row> testRdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(1, new DenseVector(new double[]{8.0d, 9.0d, 6.5d})),
				RowFactory.create(2, new DenseVector(new double[]{7.0d, 100.0d, 3.4d})),
				RowFactory.create(3, new DenseVector(new double[]{0.25d,4.0d,0.0d,1.0d,22.0d,0.0d,1.0d,0.25d}))
		));

		//[0.25,4.0,0.0,1.0,22.0,0.0,1.0,0.25]
		StructType schema1 = new StructType(new StructField[]{
				new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("features", new VectorUDT(), false, Metadata.empty()),
		});

		Dataset<Row> testData = spark.createDataFrame(testRdd, schema1);
		testData.show(false);
		Dataset<Row> rowDataset = sparkModel.transform(testData);
		rowDataset.show(false);
		List<Row> rows = rowDataset.collectAsList();
		for(Row row: rows) {
			DenseVector denseVector = (DenseVector) row.get(1);
			Map<String, Object> data = new HashMap<String, Object>();
			data.put("id", 1);
			data.put("features", denseVector.toArray());
			transformer.transform(data);
			System.out.println(data);
		}


	}

	@Test
	public void test1(){


		StructType schema = createStructType(new StructField[]{
				createStructField("id", IntegerType, false),
				createStructField("hour", IntegerType, false),
				createStructField("mobile", DoubleType, false),
				createStructField("userFeatures", new VectorUDT(), false),
				createStructField("clicked", DoubleType, false)
		});
		Row row = RowFactory.create(0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0);
		Dataset<Row> dataset = spark.createDataFrame(Arrays.asList(row), schema);

		dataset.show(false);
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"hour", "mobile", "userFeatures"})
				.setOutputCol("features");

		Dataset<Row> output = assembler.transform(dataset);
		System.out.println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column " +
				"'features'");
		output.select("features", "clicked").show(false);
	}

	@Test
	public void dummyTest(){
		int [] indices1 ={0,1,2,3,4};
		double [] values1 ={1.0,16.0,5.0,1.0,84.0};
		System.out.println(Vectors.sparse(7, indices1, values1).compressed());

		int [] indices2 ={0,4};
		double [] values2 ={1.0,16.0};
		System.out.println(Vectors.sparse(7, indices2, values2).compressed());
	}
}
