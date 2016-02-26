package com.flipkart.fdp.ml.importer;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.transformer.Transformer;
import com.google.gson.Gson;

/**
 * Created by akshay.us on 2/23/16.
 */
public class ModelImporter {
    private static final Gson gson = new Gson();

    public static Transformer importTransormer(byte[] serializedModelInfo, Class<? extends ModelInfo> modelinfo) {
        return importModelInfo(serializedModelInfo, modelinfo).getTransformer();
    }

    public static ModelInfo importModelInfo(byte[] serializedModelInfo, Class<? extends ModelInfo> modelinfo) {
        return gson.fromJson(new String(serializedModelInfo), modelinfo);
    }
}
