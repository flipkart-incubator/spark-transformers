package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.google.gson.Gson;

import java.util.HashMap;
import java.util.Map;

/**
 * Exports a {@link ModelInfo} object into byte[].
 * The serialization format currently being used is json
 * */
public class ModelExporter {
    private static final Gson gson = new Gson();


    /**
     * Exports a {@link ModelInfo} object into byte[].
     * The serialization format currently being used is json
     *
     * @return byte[]
     * @param modelInfo model info to be exported of type {@link ModelInfo}
     * */
    public static byte[] export(ModelInfo modelInfo) {
        Map<String, String> map = new HashMap<String, String>();
        map.put("_class", modelInfo.getClass().getCanonicalName());
        map.put("_model_info", gson.toJson(modelInfo));
        return gson.toJson(map).getBytes();
    }
}
