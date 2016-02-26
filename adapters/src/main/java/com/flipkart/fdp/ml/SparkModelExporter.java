package com.flipkart.fdp.ml;

import com.flipkart.fdp.ml.export.ModelExporter;

/**
 * Created by akshay.us on 2/26/16.
 */
public class SparkModelExporter {
    public static byte[]  export(Object scalaModel) {
        return ModelExporter.export(
                ModelInfoAdapterFactory.getAdapter(scalaModel.getClass())
                        .getModelInfo(scalaModel));
    }
}
