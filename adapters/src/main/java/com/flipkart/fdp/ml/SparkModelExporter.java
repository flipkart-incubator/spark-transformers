package com.flipkart.fdp.ml;

import com.flipkart.fdp.ml.export.ModelExporter;

/**
 * This class will be a single unified interface to Data Scientists to export
 * their trained model in spark to a byte[]
 */
public class SparkModelExporter {
    public static byte[]  export(Object scalaModel) {
        return ModelExporter.export(
                ModelInfoAdapterFactory.getAdapter(scalaModel.getClass())
                        .getModelInfo(scalaModel));
    }
}
