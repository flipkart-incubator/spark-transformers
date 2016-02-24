package com.flipkart.fdp.ml.importer;

import com.google.gson.Gson;

/**
 * Created by akshay.us on 2/23/16.
 */
public class ModelImporter {
    private static final Gson gson = new Gson();

    //cannot name it import as it is a reserved keyword in java
    public static<T> T importModel(String serializedModelInfo, Class<T> modelinfo) {
        return gson.fromJson(serializedModelInfo, modelinfo);
    }
}
