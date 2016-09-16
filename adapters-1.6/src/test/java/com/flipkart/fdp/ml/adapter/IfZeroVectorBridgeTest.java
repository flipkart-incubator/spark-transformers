package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.IfZeroVector;
import com.flipkart.fdp.ml.Log1PScaler;
import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.catalyst.JavaTypeInference;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Before;
import org.junit.Test;
import scala.Array;
import scala.Tuple2;
import scala.collection.mutable.WrappedArray;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class IfZeroVectorBridgeTest extends SparkTestBase implements Serializable {

    transient DataFrame denseOrderDF = null;
    transient DataFrame sparseOrderDF = null;

    transient final List<String> orderData = Arrays.asList(new String[]{
            "0.0,0.0,0.0\tLenovo Yogapad v1",
            "0.0,0.1,0.2\tNike Airmax 2015",
            "1.0,1.0,1.0\tXiaomi Redmi Note"
    });

    public static double[] str2d(String[] in) {
        double[] out =  new double[in.length];
        for(int i=0; i<in.length; i++) {
            out[i] = Double.parseDouble(in[i]);
        }
        return out;
    }

    public DataFrame createDF(JavaRDD<Tuple2<Vector, String>> rdd) {

        // Generate the schema based on the string of schema
        List<StructField> fields = new ArrayList<StructField>();
        fields.add(DataTypes.createStructField("vectorized_count", new VectorUDT(), true));
        fields.add(DataTypes.createStructField("product_title", DataTypes.StringType, true));

        StructType schema = DataTypes.createStructType(fields);
        // Convert records of the RDD (people) to Rows.
        JavaRDD<Row> rowRDD = rdd.map(
                new Function<Tuple2<Vector, String>, Row>() {
                    public Row call(Tuple2<Vector, String> record) {
                        return RowFactory.create(record._1(), record._2());
                    }
                });

        return sqlContext.createDataFrame(rowRDD, schema);
    }

    @Before
    public void before() {


        JavaRDD<String> orderTextRdd = sc.parallelize(orderData);
        JavaRDD<Tuple2<double[], String>> orderRdd = orderTextRdd.map(new Function<String, Tuple2<double[], String>>() {

            @Override
            public Tuple2<double[], String> call(String v1) throws Exception {
                String[] fields = v1.split("\t");
                String[] arr = fields[0].split(",");
                double[] d = str2d(arr);
                return new Tuple2<double[], String>(d, fields[1]);
            }
        });

        JavaRDD<Tuple2<Vector, String>> denseOrderRdd = orderRdd.map(
                new Function<Tuple2<double[], String>, Tuple2<Vector, String>>() {
                    @Override
                    public Tuple2<Vector, String> call(Tuple2<double[], String> t) throws Exception {
                        return new Tuple2<Vector, String>(new DenseVector(t._1()), t._2());
                    }
                });

        JavaRDD<Tuple2<Vector, String>> sparseOrderRdd = orderRdd.map(
                new Function<Tuple2<double[], String>, Tuple2<Vector, String>>() {
                    @Override
                    public Tuple2<Vector, String> call(Tuple2<double[], String> t) throws Exception {
                        return new Tuple2<Vector, String>(new SparseVector(5, new int[]{0,1,2}, t._1()), t._2());
                    }
                });

        this.denseOrderDF = createDF(denseOrderRdd).persist();
        this.sparseOrderDF = createDF(sparseOrderRdd).persist();
    }

    @Test
    public void testIfZeroVectorDense() {
        IfZeroVector sparkModel = new IfZeroVector()
                .setInputCol("vectorized_count")
                .setOutputCol("product_title_filtered")
                .setThenSetValue("others")
                .setElseSetCol("product_title");

        //compare predictions
        Row[] sparkOutput = sparkModel.transform(denseOrderDF).orderBy("product_title").select("product_title_filtered").collect();
        assertEquals("others", sparkOutput[0].get(0));
        assertEquals("Nike Airmax 2015", sparkOutput[1].get(0));
        assertEquals("Xiaomi Redmi Note", sparkOutput[2].get(0));
    }

    @Test
    public void testIfZeroVectorSparse() {
        IfZeroVector sparkModel = new IfZeroVector()
                .setInputCol("vectorized_count")
                .setOutputCol("product_title_filtered")
                .setThenSetValue("others")
                .setElseSetCol("product_title");
        System.out.println(sparseOrderDF.schema());
        DataFrame transformed = sparkModel.transform(sparseOrderDF).orderBy("product_title");
        System.out.println(transformed.schema());
        //compare predictions
        Row[] sparkOutput = transformed.select("product_title_filtered").collect();
        assertEquals("others", sparkOutput[0].get(0));
        assertEquals("Nike Airmax 2015", sparkOutput[1].get(0));
        assertEquals("Xiaomi Redmi Note", sparkOutput[2].get(0));
    }

    @Test
    public void testModelExportAndImportDense() {
        IfZeroVector sparkModel = new IfZeroVector()
                .setInputCol("vectorized_count")
                .setOutputCol("product_title_filtered")
                .setThenSetValue("others")
                .setElseSetCol("product_title");
        byte[] exportedModel = ModelExporter.export(sparkModel, denseOrderDF);
        final Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        //compare predictions
        Row[] denseOrder = denseOrderDF.collect();

        for (int i = 0; i < denseOrder.length; i++) {
            double[] input = ((Vector) denseOrder[i].get(0)).toArray();
            System.out.println("Input double array from dense = " + Arrays.toString(input));
            String colValue = ((String) denseOrder[i].get(1));

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("vectorized_count", input);
            data.put("product_title", colValue);

            transformer.transform(data);
            double[] tInput = (double[]) data.get("vectorized_count");
            String tColValue = (String) data.get("product_title");
            String output = (String) data.get("product_title_filtered");
            assertEquals(input, tInput);
            assertEquals(colValue, tColValue);
            String expectedOutput = (i == 0) ? "others" : colValue;
            System.out.println(output);
            assertEquals(expectedOutput, output);
        }
    }

    @Test
    public void testModelExportAndImportSparse() {
        IfZeroVector sparkModel = new IfZeroVector()
                .setInputCol("vectorized_count")
                .setOutputCol("product_title_filtered")
                .setThenSetValue("others")
                .setElseSetCol("product_title");
        byte[] exportedModel = ModelExporter.export(sparkModel, sparseOrderDF);
        final Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);
        //compare predictions
        Row[] sparseOrder = sparseOrderDF.collect();

        for (int i = 0; i < sparseOrder.length; i++) {
            double[] input = ((Vector) sparseOrder[i].get(0)).toArray();
            System.out.println("Input double array from sparse = " + Arrays.toString(input));
            String colValue = ((String) sparseOrder[i].get(1));

            Map<String, Object> data = new HashMap<String, Object>();
            data.put("vectorized_count", input);
            data.put("product_title", colValue);

            transformer.transform(data);
            double[] tInput = (double[]) data.get("vectorized_count");
            String tColValue = (String) data.get("product_title");
            String output = (String) data.get("product_title_filtered");
            assertEquals(input, tInput);
            assertEquals(colValue, tColValue);
            String expectedOutput = (i == 0) ? "others" : colValue;
            System.out.println(output);
            assertEquals(expectedOutput, output);
        }
    }

}
