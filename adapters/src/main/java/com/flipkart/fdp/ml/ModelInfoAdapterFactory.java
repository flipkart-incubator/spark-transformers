package com.flipkart.fdp.ml;

import com.flipkart.fdp.ml.adapter.DecisionTreeModelInfoInfoAdapter;
import com.flipkart.fdp.ml.adapter.LogisticRegressionModelInfoInfoAdapter;
import com.flipkart.fdp.ml.adapter.ModelInfoAdapter;
import com.flipkart.fdp.ml.adapter.RandomForestModelInfoInfoAdapter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Created by akshay.us on 2/25/16.
 */
public class ModelInfoAdapterFactory {
    private static final Logger LOG = LoggerFactory.getLogger(ModelInfoAdapterFactory.class);

    private static final Map<String, ModelInfoAdapter> registry = new java.util.HashMap<String, ModelInfoAdapter>();

    static {
        register(new LogisticRegressionModelInfoInfoAdapter());
        register(new DecisionTreeModelInfoInfoAdapter());
        register(new RandomForestModelInfoInfoAdapter());
    }

    private static void register(ModelInfoAdapter adapter) {
        registry.put(adapter.getSource().getCanonicalName() + "/" + adapter.getTarget().getCanonicalName(), adapter);
        registry.put(adapter.getSource().getCanonicalName(), adapter);
    }

    public static ModelInfoAdapter getAdapter(Class from) {
        return registry.get(from.getCanonicalName());
    }

    public static ModelInfoAdapter getAdapter(Class from, Class to) {
        return registry.get(from.getCanonicalName() + "/" + to.getCanonicalName());
    }
}
