package com.flipkart.fdp.ml.modelinfo;

import java.util.ArrayList;

public class RandomForestModelInfo implements ModelInfo {
    public String algorithm;
    public ArrayList<DecisionTreeModelInfo> trees = new ArrayList<DecisionTreeModelInfo>();

    public RandomForestModelInfo() {
    }


}
