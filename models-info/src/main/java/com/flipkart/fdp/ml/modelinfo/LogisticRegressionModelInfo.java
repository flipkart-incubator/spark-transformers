package com.flipkart.fdp.ml.modelinfo;

import lombok.ToString;

@ToString
public class LogisticRegressionModelInfo implements ModelInfo{
    public double[] weights;
    public double intercept;
    public int numClasses;
    public int numFeatures;

}
