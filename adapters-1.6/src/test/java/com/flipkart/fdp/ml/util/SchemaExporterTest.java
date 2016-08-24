package com.flipkart.fdp.ml.util;

import com.flipkart.fdp.ml.utils.SchemaExporter;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.types.*;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;

/**
 * Created by akshay.us on 8/10/16.
 */
public class SchemaExporterTest {

    /**
     * Output :
     [{"name":"id","datatype":"double"},{"name":"feature","datatype":"double"}]
     * */
    @Test
    public void testSchema() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("feature", DataTypes.DoubleType, false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportSchemaToJson(schema));
    }

    /**
     * Output :
     [{"name":"id","datatype":"double"},{"name":"label","datatype":"double"},{"name":"features","datatype":"double []"}]
     * */
    @Test
    public void testSchema1() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportSchemaToJson(schema));
    }

    /**
     * Output :
     [{"name":"id","datatype":"double"},{"name":"text","datatype":"String []"}]
     * */
    @Test
    public void testSchema2() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportSchemaToJson(schema));
    }

    /**
     * Output :
     [{"name":"id","datatype":"double"},{"name":"value1","datatype":"double"},{"name":"vector1","datatype":"double []"}]
     * */
    @Test
    public void testSchema3() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("value1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("vector1", new VectorUDT(), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportSchemaToJson(schema));
    }

    /**
     * Output :
     [{"name":"feature","datatype":"double"},{"name":"id","datatype":"double"}]
     * */
    @Test
    public void testColumnExport() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("feature", DataTypes.DoubleType, false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(new HashSet<String>(Arrays.asList("id", "feature")),schema));
    }

    /**
     * Output :
     [{"name":"features","datatype":"double []"},{"name":"id","datatype":"double"}]
     * */
    @Test
    public void testColumnExport1() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(new HashSet<String>(Arrays.asList("id", "features")),schema));
    }

    /**
     * Output :
     [{"name":"id","datatype":"double"},{"name":"text","datatype":"String []"}]
     * */
    @Test
    public void testColumnExport2() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(new HashSet<String>(Arrays.asList("id", "text")),schema));
    }

    /**
     * Output :
     [{"name":"id","datatype":"double"},{"name":"vector1","datatype":"double []"}]
     * */
    @Test
    public void testColumnExport3() {
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("value1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("vector1", new VectorUDT(), false, Metadata.empty())
        });
        System.out.println(SchemaExporter.exportToJson(new HashSet<String>(Arrays.asList("id", "vector1")),schema));
    }
}
