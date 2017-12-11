package com.flipkart.fdp.ml.transformer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.flipkart.fdp.ml.modelinfo.DecisionTreeModelInfo;
import com.flipkart.fdp.ml.modelinfo.GradientBoostModelInfo;
import com.github.fommil.netlib.BLAS;

public class GradientBoostClassificationTransformer implements Transformer {
    private final GradientBoostModelInfo forest;
    private final List<DecisionTreeTransformer> subTransformers;

    public GradientBoostClassificationTransformer(final GradientBoostModelInfo forest) {
        this.forest = forest;
        this.subTransformers = new ArrayList<>(forest.getTrees().size());
        for (final DecisionTreeModelInfo tree : forest.getTrees()) {
            subTransformers.add((DecisionTreeTransformer) tree.getTransformer());
        }
    }

    public double predict(final double[] input) {
        double[] treePredictions = new double[subTransformers.size()];
        double [] treeWeights = new double[subTransformers.size()];
        List<Double> modelTreeWeights = forest.getTreeWeights();
        int index = 0;
        for (final DecisionTreeTransformer treeTransformer : subTransformers) {
        	treePredictions[index] = treeTransformer.predict(input);
        	treeWeights[index] = modelTreeWeights.get(index);
        	index++;
        }
		double prediction = BLAS.getInstance().ddot(subTransformers.size(), treePredictions, 1, treeWeights, 1);
		if (prediction > 0.0)
			return 1.0;
		else
			return 0.0;
    }

    @Override
    public void transform(Map<String, Object> input) {
        double[] inp = (double[]) input.get(forest.getInputKeys().iterator().next());
        input.put(forest.getOutputKeys().iterator().next(), predict(inp));
    }


    @Override
    public Set<String> getInputKeys() {
        return forest.getInputKeys();
    }

    @Override
    public Set<String> getOutputKeys() {
        return forest.getOutputKeys();
    }

}
