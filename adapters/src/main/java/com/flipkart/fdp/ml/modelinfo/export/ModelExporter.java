package com.flipkart.fdp.ml.modelinfo.export;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.google.gson.Gson;

/**
 * Created by akshay.us on 2/19/16.
 */
public class ModelExporter {
    public String export(ModelInfo modelInfo) {
        return new Gson().toJson(modelInfo);
    }
}
