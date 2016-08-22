package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.adapter.SparkTestBase;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.DataFrame;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class LogisticRegression1ExporterTest extends SparkTestBase {

    @Test
    public void shouldExportAndImportCorrectly() {
        //prepare data
        String datapath = "src/test/resources/binary_classification_test.libsvm";

        DataFrame trainingData = sqlContext.read().format("libsvm").load(datapath);

        //Train model in spark
        LogisticRegressionModel lrmodel = new LogisticRegression().fit(trainingData);

        //Export this model
        byte[] exportedModel = ModelExporter.export(lrmodel, trainingData);

        //Import it back
        LogisticRegressionModelInfo importedModel = (LogisticRegressionModelInfo) ModelImporter.importModelInfo(exportedModel);

        //check if they are exactly equal with respect to their fields
        //it maybe edge cases eg. order of elements in the list is changed
        assertEquals(lrmodel.intercept(), importedModel.getIntercept(), 0.01);
        assertEquals(lrmodel.numClasses(), importedModel.getNumClasses(), 0.01);
        assertEquals(lrmodel.numFeatures(), importedModel.getNumFeatures(), 0.01);
        assertEquals(lrmodel.getThreshold(), importedModel.getThreshold(), 0.01);
        for (int i = 0; i < importedModel.getNumFeatures(); i++)
            assertEquals(lrmodel.weights().toArray()[i], importedModel.getWeights()[i], 0.01);

        assertEquals(lrmodel.getFeaturesCol(), importedModel.getInputKeys().iterator().next());
        assertEquals(lrmodel.getPredictionCol(), importedModel.getOutputKeys().iterator().next());
    }
}
