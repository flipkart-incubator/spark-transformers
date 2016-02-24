package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.google.gson.Gson;

/**
 * Created by akshay.us on 2/19/16.
 */
public class ModelExporter {
    private static final Gson gson = new Gson();

    public static String export(ModelInfo modelInfo) {
        return gson.toJson(modelInfo);
    }
}
