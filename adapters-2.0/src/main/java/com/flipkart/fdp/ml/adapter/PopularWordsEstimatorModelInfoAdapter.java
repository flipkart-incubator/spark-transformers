package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.PopularWordsEstimatorModelInfo;
import org.apache.spark.ml.PopularWordsModel;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

public class PopularWordsEstimatorModelInfoAdapter extends AbstractModelInfoAdapter<PopularWordsModel, PopularWordsEstimatorModelInfo> {

	@Override
	PopularWordsEstimatorModelInfo getModelInfo(PopularWordsModel from) {
		PopularWordsEstimatorModelInfo modelInfo = new PopularWordsEstimatorModelInfo();
		modelInfo.setPopularWords(new HashSet<>(Arrays.asList(from.popularWords())));

		Set<String> inputKeys = new LinkedHashSet<>();
		inputKeys.add(from.getInputCol());
		modelInfo.setInputKeys(inputKeys);

		Set<String> outputKeys = new LinkedHashSet<>();
		outputKeys.add(from.getOutputCol());
		modelInfo.setOutputKeys(outputKeys);

		return modelInfo;
	}

	@Override
	public Class<PopularWordsModel> getSource() {
		return PopularWordsModel.class;
	}

	@Override
	public Class<PopularWordsEstimatorModelInfo> getTarget() {
		return PopularWordsEstimatorModelInfo.class;
	}
}
