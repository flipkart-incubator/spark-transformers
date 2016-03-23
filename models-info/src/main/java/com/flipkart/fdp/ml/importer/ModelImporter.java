package com.flipkart.fdp.ml.importer;

import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;
import com.flipkart.fdp.ml.transformer.Transformer;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.util.Map;

/**
 * Imports byte[] representing a model into corresponding {@link ModelInfo} object.
 * The serialization format currently being used is json
 */
public class ModelImporter {
    private static final Gson gson = new Gson();


    /**
     * Imports byte[] representing a model into corresponding {@link ModelInfo} object
     * and returns the transformer for this model.
     *
     * @param serializedModelInfo byte[] representing the serialized data
     * @return transformer for the imported model of type {@link Transformer}
     */
    public static Transformer importAndGetTransformer(byte[] serializedModelInfo) {
        return importModelInfo(serializedModelInfo).getTransformer();
    }

    /**
     * Imports byte[] representing a model into corresponding {@link ModelInfo} object.
     * The serialization format currently being used is json
     *
     * @param serializedModelInfo byte[] representing the serialized data
     * @return model info imported of type {@link ModelInfo}
     */
    public static ModelInfo importModelInfo(byte[] serializedModelInfo) {
        String data = new String(serializedModelInfo, SerializationConstants.CHARSET);
        Map<String, String> map = gson.fromJson(data, new TypeToken<Map<String, String>>() {
        }.getType());
        Class modelClass = null;
        try {
            modelClass = Class.forName(map.get(SerializationConstants.TYPE_IDENTIFIER));
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        if (modelClass == PipelineModelInfo.class) {
            String[] serializedModelInfos = gson.fromJson(map.get(SerializationConstants.MODEL_INFO_IDENTIFIER), String[].class);
            ModelInfo[] modelInfos = new ModelInfo[serializedModelInfos.length];
            for (int i = 0; i < modelInfos.length; i++) {
                modelInfos[i] = importModelInfo(serializedModelInfos[i].getBytes());
            }
            PipelineModelInfo pipelineModelInfo = new PipelineModelInfo();
            pipelineModelInfo.setStages(modelInfos);
            return pipelineModelInfo;
        } else {
            return (ModelInfo) gson.fromJson(map.get(SerializationConstants.MODEL_INFO_IDENTIFIER), modelClass);
        }
    }
}
