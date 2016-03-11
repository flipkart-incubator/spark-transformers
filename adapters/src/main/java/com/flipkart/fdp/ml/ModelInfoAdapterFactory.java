package com.flipkart.fdp.ml;

import com.flipkart.fdp.ml.adapter.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * A factory that will create and cache various adapters of type {@link ModelInfoAdapter}
 * The purpose of the class is to abstract away that logic of identifying which adapter to use.
 */
public class ModelInfoAdapterFactory {
    private static final Logger LOG = LoggerFactory.getLogger(ModelInfoAdapterFactory.class);

    private static final Map<String, ModelInfoAdapter> registry = new java.util.HashMap<String, ModelInfoAdapter>();

    static {
        register(new LogisticRegressionModelInfoAdapter());
        register(new LogisticRegressionModelInfoAdapter1());
        register(new DecisionTreeModelInfoAdapter());
        register(new RandomForestModelInfoAdapter());
        register(new StringIndexerModelInfoAdapter());
        register(new HashingTFModelInfoAdapter());
        register(new OneHotEncoderModelInfoAdapter());
    }

    private static void register(ModelInfoAdapter adapter) {
        registry.put(adapter.getSource().getCanonicalName() + "/" + adapter.getTarget().getCanonicalName(), adapter);
        registry.put(adapter.getSource().getCanonicalName(), adapter);
    }

    /**
     * Returns the respective {@link ModelInfoAdapter} instance that will adapt for the input model class
     *
     * @param from The model class that needs to be adapted.
     * @return The respective {@link ModelInfoAdapter} instance
     */
    public static ModelInfoAdapter getAdapter(Class from) {
        return registry.get(from.getCanonicalName());
    }

    /**
     * Returns the respective {@link ModelInfoAdapter} instance that will adapt for the input model class.
     * In case more than one adapters are available for a model the target to class should be specified
     * to fetch that specific adapter
     *
     * @param from The model class that needs to be adapted.
     * @param to   The {@link com.flipkart.fdp.ml.modelinfo.ModelInfo} class that has to be adapted to
     * @return The respective {@link ModelInfoAdapter} instance
     */
    public static ModelInfoAdapter getAdapter(Class from, Class to) {
        return registry.get(from.getCanonicalName() + "/" + to.getCanonicalName());
    }
}
