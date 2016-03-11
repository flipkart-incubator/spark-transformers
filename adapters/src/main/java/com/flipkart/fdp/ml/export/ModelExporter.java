package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.ModelInfoAdapterFactory;
import com.flipkart.fdp.ml.importer.SerializationConstants;
import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.google.gson.Gson;
import org.apache.spark.sql.DataFrame;

import java.util.HashMap;
import java.util.Map;

/**
 * Exports a {@link ModelInfo} object into byte[].
 * The serialization format currently being used is json
 */
public class ModelExporter {
    private static final Gson gson = new Gson();

    /**
     * Exports a Model object into byte[].
     * The serialization format currently being used is json
     *
     * @param model model info to be exported
     * @param df Data frame that is used for training is required for some models as state information is being stored as column metadata by some spark models
     * @return byte[]
     */
    public static byte[] export(Object model, DataFrame df) {
        return ModelExporter.export(
                ModelInfoAdapterFactory.getAdapter(model.getClass())
                        .getModelInfo(model, df));
    }

    /**
     * Exports a {@link ModelInfo} object into byte[].
     * The serialization format currently being used is json
     *
     * @param modelInfo model info to be exported of type {@link ModelInfo}
     * @return byte[]
     */
    private static byte[] export(ModelInfo modelInfo) {
        Map<String, String> map = new HashMap<String, String>();
        map.put(SerializationConstants.TYPE_IDENTIFIER, modelInfo.getClass().getCanonicalName());
        map.put(SerializationConstants.MODEL_INFO_IDENTIFIER, gson.toJson(modelInfo));
        return gson.toJson(map).getBytes();
    }
}
