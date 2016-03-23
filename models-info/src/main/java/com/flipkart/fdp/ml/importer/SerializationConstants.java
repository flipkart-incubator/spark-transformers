package com.flipkart.fdp.ml.importer;

/**
 * Class holding constants used in serialization
 */
public class SerializationConstants {
    //key to identify type in serialized format
    public static final String TYPE_IDENTIFIER = "_class";
    //key to identify model info payload in serialized format
    public static final String MODEL_INFO_IDENTIFIER = "_model_info";
    //key to identify the spark version it was imported from
    public static final String SPARK_VERSION="_spark_version";
    //key to identify the exporter library version it was exported with
    public static final String EXPORTER_LIBRARY_VERSION="_exporter_library_version";
}
