package com.flipkart.fdp.ml.adapter;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.junit.Test;

import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;

/**
 * Created by akshay.us on 8/29/16.
 */
public class GradientBoostClassificationModelInfoAdapterTest extends SparkTestBase {

	@Test
	public void testRandomForestClassification() {
		// Load the data stored in LIBSVM format as a DataFrame.
		String datapath = "src/test/resources/binary_classification_test.libsvm";

		Dataset<Row> data = spark.read().format("libsvm").load(datapath);

		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];

		// Train a RandomForest model.
		GBTClassificationModel classificationModel = new GBTClassifier().fit(trainingData);

		byte[] exportedModel = ModelExporter.export(classificationModel);

		Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

		List<Row> sparkOutput =
		        classificationModel.transform(testData).select("features", "prediction").collectAsList();

		// compare predictions
		for (Row row : sparkOutput) {
			Map<String, Object> data_ = new HashMap<>();
			data_.put("features", ((SparseVector) row.get(0)).toArray());
			transformer.transform(data_);
			System.out.println(data_);
			System.out.println(data_.get("prediction")+" ,"+row.get(1));
			assertEquals((double) data_.get("prediction"), (double) row.get(1), EPSILON);
		}

	}
}
