package com.flipkart.fdp.ml.adapter;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.junit.After;
import org.junit.Before;

import java.io.IOException;

/**
 * Base class for test that need to create and use a spark context.
 */
public class SparkTestBase {
    protected transient SparkSession spark;
    protected transient JavaSparkContext jsc;

    @Before
    public void setUp() throws IOException {
        spark = SparkSession.builder()
                .master("local[2]")
                .appName(getClass().getSimpleName())
                .getOrCreate();
        jsc = new JavaSparkContext(spark.sparkContext());
    }

    @After
    public void tearDown() {
        spark.stop();
        spark = null;
    }

    /**
     * An alias for RowFactory.create.
     */
    public Row cr(Object... values) {
        return RowFactory.create(values);
    }
}
