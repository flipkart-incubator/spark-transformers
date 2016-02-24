package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.adapter.LogisticRegressionModelInfoInfoAdapter;
import com.flipkart.fdp.ml.adapter.SparkTestBase;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
public class LogisticRegressionExporterTest extends SparkTestBase {

    @Test
    public void shouldExportAndImportCorrectly() {
        String datapath = "src/test/resources/binary_classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());
        LogisticRegressionModelInfoInfoAdapter logisticRegressionBridgeIn =
                new LogisticRegressionModelInfoInfoAdapter();

        LogisticRegressionModelInfo logisticRegressionModelInfo =
                logisticRegressionBridgeIn.transform(lrmodel);

        //Export this model
        String exportedModel = ModelExporter.export(logisticRegressionModelInfo);

        //Import it back
        LogisticRegressionModelInfo importedModel = ModelImporter.importModel(exportedModel, LogisticRegressionModelInfo.class);

        //check if they are exactly equal
        assertEquals(lrmodel.intercept(), importedModel.intercept, 0.01);
        assertEquals(lrmodel.numClasses(), importedModel.numClasses, 0.01);
        assertEquals(lrmodel.numFeatures(), importedModel.numFeatures, 0.01);

        for( int i = 0 ; i <importedModel.numFeatures; i++)
            assertEquals(lrmodel.weights().toArray()[i], importedModel.weights[i], 0.01);

    }
}
