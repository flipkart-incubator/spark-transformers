package com.flipkart.fdp.ml.importer;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.transformer.Transformer;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.util.Map;

/**
 * Imports byte[] representing a model into corresponding {@link ModelInfo} object.
 * The serialization format currently being used is json
 * */
public class ModelImporter {
    private static final Gson gson = new Gson();


    /**
     * Imports byte[] representing a model into corresponding {@link ModelInfo} object
     * and returns the transformer for this model.
     *
     * @return transformer for the imported model of type {@link Transformer}
     * @param serializedModelInfo byte[] representing the serialized data
     * */
    public static Transformer importAndGetTransformer(byte[] serializedModelInfo) {
        return importModelInfo(serializedModelInfo).getTransformer();
    }

    /**
     * Imports byte[] representing a model into corresponding {@link ModelInfo} object.
     * The serialization format currently being used is json
     *
     * @return model info imported of type {@link ModelInfo}
     * @param serializedModelInfo byte[] representing the serialized data
     * */
    public static ModelInfo importModelInfo(byte[] serializedModelInfo) {
        String data = new String(serializedModelInfo);
        Map<String, String> map = gson.fromJson(data, new TypeToken<Map<String, String>>(){}.getType());
        Class modelClass = null;
        try {
            modelClass = Class.forName(map.get("_class"));
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        return (ModelInfo)gson.fromJson(map.get("_model_info"), modelClass);
    }
}
