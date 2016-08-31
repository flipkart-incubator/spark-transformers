package com.flipkart.fdp.ml.adapter;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.junit.After;
import org.junit.Before;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for test that need to create and use a spark context.
 */
public class SparkTestBase {
    private static final Logger LOG = LoggerFactory.getLogger(SparkTestBase.class);
    protected JavaSparkContext sc;
    protected SQLContext sqlContext;
    public static final double EPSILON = 1.0e-6;

    @Before
    public void setup() {
        SparkConf sparkConf = new SparkConf();
        String master = "local[2]";
        sparkConf.setMaster(master);
        sparkConf.setAppName("Local Spark Unit Test");
        sc = new JavaSparkContext(new SparkContext(sparkConf));
        sqlContext = new SQLContext(sc);
    }

    @After
    public void tearDown() {
        sc.close();
        sqlContext = null;
    }

    /**
     * An alias for RowFactory.create.
     */
    public Row cr(Object... values) {
        return RowFactory.create(values);
    }
}
