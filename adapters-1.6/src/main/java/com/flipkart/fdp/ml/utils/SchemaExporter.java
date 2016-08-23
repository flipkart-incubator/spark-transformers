package com.flipkart.fdp.ml.utils;

import com.google.gson.Gson;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.types.*;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by akshay.us on 8/10/16.
 */
public class SchemaExporter {
    private static final String DOUBLE = "double";
    private static final String DOUBLE_ARRAY = "double []";
    private static final String BOOLEAN = "boolean";
    private static final String STRING = "String";
    private static final String STRING_ARRAY = "String []";

    private static final Gson gson = new Gson();


    public static String exportToJson(Set<String> columns, StructType dfSchema) {
        //This would contain column name along with type of a dataframe

        Map<String, String> schema = new LinkedHashMap<>();

        for (String column : columns) {
            StructField field = dfSchema.fields()[ dfSchema.fieldIndex(column) ];

            if (field.dataType() instanceof StringType) {
                schema.put(field.name(), STRING);
            } else if (field.dataType() instanceof BooleanType) {
                schema.put(field.name(), BOOLEAN);
            } else if (field.dataType() instanceof VectorUDT) {
                schema.put(field.name(), DOUBLE_ARRAY);
            } else if (field.dataType() instanceof DoubleType || field.dataType() instanceof DecimalType || field.dataType() instanceof FloatType ||
                    field.dataType() instanceof IntegerType || field.dataType() instanceof LongType || field.dataType() instanceof ShortType) {
                schema.put(field.name(), DOUBLE);
            } else if (field.dataType() instanceof ArrayType) {
                if(((ArrayType)field.dataType()).elementType() instanceof StringType) {
                    schema.put(field.name(), STRING_ARRAY);
                }else if(((ArrayType)field.dataType()).elementType() instanceof DoubleType) {
                    schema.put(field.name(), DOUBLE_ARRAY);
                }else {
                    throw new UnsupportedOperationException("Cannot support data of type " + field.dataType());
                }
            }
            else {
                throw new UnsupportedOperationException("Cannot support data of type " + field.dataType());
            }
        }
        return gson.toJson(schema);
    }


    public static String exportSchemaToJson(StructType dfSchema) {
        //This would contain column name along with type of a dataframe

        Map<String, String> schema = new LinkedHashMap<>();

        for (StructField field : dfSchema.fields()) {
            if (field.dataType() instanceof StringType) {
                schema.put(field.name(), STRING);
            } else if (field.dataType() instanceof BooleanType) {
                schema.put(field.name(), BOOLEAN);
            } else if (field.dataType() instanceof VectorUDT) {
                schema.put(field.name(), DOUBLE_ARRAY);
            } else if (field.dataType() instanceof DoubleType || field.dataType() instanceof DecimalType || field.dataType() instanceof FloatType ||
                    field.dataType() instanceof IntegerType || field.dataType() instanceof LongType || field.dataType() instanceof ShortType) {
                schema.put(field.name(), DOUBLE);
            } else if (field.dataType() instanceof ArrayType) {
                if(((ArrayType)field.dataType()).elementType() instanceof StringType) {
                    schema.put(field.name(), STRING_ARRAY);
                }else if(((ArrayType)field.dataType()).elementType() instanceof DoubleType) {
                    schema.put(field.name(), DOUBLE_ARRAY);
                }else {
                    throw new UnsupportedOperationException("Cannot support data of type " + field.dataType());
                }
            }
            else {
                throw new UnsupportedOperationException("Cannot support data of type " + field.dataType());
            }
        }
        return gson.toJson(schema);
    }
}
