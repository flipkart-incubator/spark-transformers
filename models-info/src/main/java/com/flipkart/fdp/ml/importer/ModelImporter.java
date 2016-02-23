package com.flipkart.fdp.ml.importer;

import com.google.gson.Gson;

/**
 * Created by akshay.us on 2/23/16.
 */
public class ModelImporter<T> {
    public T export(String serializedModelInfo, Class<T> modelinfo) {
        return new Gson().fromJson(serializedModelInfo, modelinfo);
    }
}
