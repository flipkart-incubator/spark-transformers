package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.adapter.SparkTestBase;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.modelinfo.LogisticRegressionModelInfo;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

;

public class LogisticRegressionExporterTest extends SparkTestBase {

    @Test
    public void shouldExportAndImportCorrectly() {
        String datapath = "src/test/resources/binary_classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();

        //Train model in spark
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());

        //Export this model
        byte[] exportedModel = ModelExporter.export(lrmodel);

        System.out.println(new String(exportedModel));

        //Import it back
        LogisticRegressionModelInfo importedModel = (LogisticRegressionModelInfo) ModelImporter.importModelInfo(exportedModel);

        //check if they are exactly equal with respect to their fields
        //it maybe edge cases eg. order of elements in the list is changed
        assertEquals(lrmodel.intercept(), importedModel.getIntercept(), 0.01);
        assertEquals(lrmodel.numClasses(), importedModel.getNumClasses(), 0.01);
        assertEquals(lrmodel.numFeatures(), importedModel.getNumFeatures(), 0.01);
        for (int i = 0; i < importedModel.getNumFeatures(); i++)
            assertEquals(lrmodel.weights().toArray()[i], importedModel.getWeights()[i], 0.01);

    }
}
