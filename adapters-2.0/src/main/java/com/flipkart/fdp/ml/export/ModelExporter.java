package com.flipkart.fdp.ml.export;

import com.flipkart.fdp.ml.ModelInfoAdapterFactory;
import com.flipkart.fdp.ml.importer.SerializationConstants;
import com.flipkart.fdp.ml.modelinfo.ModelInfo;
import com.flipkart.fdp.ml.modelinfo.PipelineModelInfo;
import com.flipkart.fdp.ml.utils.Constants;
import com.google.gson.Gson;

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
     * @return byte[]
     */
    public static byte[] export(Object model) {
        return export(
                ModelInfoAdapterFactory.getAdapter(model.getClass())
                        .adapt(model)).getBytes(SerializationConstants.CHARSET);
    }

    /**
     * Exports a {@link ModelInfo} object into byte[].
     * The serialization format currently being used is json
     *
     * @param modelInfo model info to be exported of type {@link ModelInfo}
     * @return byte[]
     */
    private static String export(ModelInfo modelInfo) {
        final Map<String, String> map = new HashMap<String, String>();
        map.put(SerializationConstants.SPARK_VERSION, Constants.SUPPORTED_SPARK_VERSION_PREFIX);
        map.put(SerializationConstants.EXPORTER_LIBRARY_VERSION, Constants.LIBRARY_VERSION);
        map.put(SerializationConstants.TYPE_IDENTIFIER, modelInfo.getClass().getCanonicalName());
        if (modelInfo instanceof PipelineModelInfo) {
            //custom serialization is needed as type is not encoded into gson serialized modelInfo
            PipelineModelInfo pipelineModelInfo = (PipelineModelInfo) modelInfo;
            String[] serializedModels = new String[pipelineModelInfo.getStages().length];
            for (int i = 0; i < serializedModels.length; i++) {
                serializedModels[i] = export(pipelineModelInfo.getStages()[i]);
            }
            map.put(SerializationConstants.MODEL_INFO_IDENTIFIER, gson.toJson(serializedModels));
        } else {
            map.put(SerializationConstants.MODEL_INFO_IDENTIFIER, gson.toJson(modelInfo));
        }
        return gson.toJson(map);
    }
}
