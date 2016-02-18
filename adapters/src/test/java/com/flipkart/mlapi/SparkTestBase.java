package com.flipkart.mlapi;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Before;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SparkTestBase {
    private static final Logger LOG = LoggerFactory.getLogger(SparkTestBase.class);
    protected JavaSparkContext sc;

    @Before
    public void setup() {
        SparkConf sparkConf = new SparkConf();
        String master = "local[1]";
        sparkConf.setMaster(master);
        sparkConf.setAppName("Local Spark Unit Test");
        sc = new JavaSparkContext(new SparkContext(sparkConf));
    }

    @After
    public void tearDown() {
        sc.close();
    }
}
