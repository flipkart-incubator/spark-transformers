package com.flipkart.fdp.ml;

import com.flipkart.fdp.ml.adapter.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.PopularWordsEstimator;

import java.util.Map;

/**
 * A factory that will create and cache various adapters of type {@link ModelInfoAdapter}
 * The purpose of the class is to abstract away that logic of identifying which adapter to use.
 */
@Slf4j
public class ModelInfoAdapterFactory {

    private static final Map<String, ModelInfoAdapter> registry = new java.util.HashMap<>();

    static {
        register(new LogisticRegressionModelInfoAdapter());
        register(new LogisticRegressionModelInfoAdapter1());
        register(new StringIndexerModelInfoAdapter());
        register(new HashingTFModelInfoAdapter());
        register(new RegexTokenizerModelInfoAdapter());
        register(new CountVectorizerModelInfoAdapter());
        register(new StandardScalerModelInfoAdapter());
        register(new MinMaxScalerModelInfoAdapter());
        register(new BucketizerModelInfoAdapter());
        register(new PipelineModelInfoAdapter());
        register(new VectorAssemblerModelAdapter());
        register(new ChiSqSelectorModelInfoAdapter());
        register(new StringMergeInfoAdapter());
        register(new StringSanitizerModelInfoAdapter());
        register(new CommonAddressFeaturesModelInfoAdapter());
        register(new PopularWordsEstimatorModelInfoAdapter());

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
