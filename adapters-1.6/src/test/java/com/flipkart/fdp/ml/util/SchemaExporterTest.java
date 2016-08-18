package com.flipkart.fdp.ml.util;

import com.flipkart.fdp.ml.utils.SchemaExporter;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.types.*;
import org.junit.Test;

/**
 * Created by akshay.us on 8/10/16.
 */
public class SchemaExporterTest {

    /**
     * Output :
     {"id":"double","label":"double","features":"double []"}
     * */
    @Test
    public void testSchema() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("feature", DataTypes.DoubleType, false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(schema));
    }
    /**
     * Output :
     {"id":"double","text":"String []"}
     * */
    @Test
    public void testSchema1() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(schema));
    }
    /**
     * Output :
     {"id":"double","value1":"double","vector1":"double []"}
     * */
    @Test
    public void testSchema2() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(schema));
    }
    /**
     * Output :
     {"id":"double","feature":"double"}
     * */
    @Test
    public void testSchema3() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("value1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("vector1", new VectorUDT(), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(schema));
    }
}
