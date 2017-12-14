package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.PopularWordsEstimatorModelInfo;

import java.util.*;

public class PopularWordsEstimatorTransformer implements Transformer {
	private PopularWordsEstimatorModelInfo modelInfo;

	public PopularWordsEstimatorTransformer(final PopularWordsEstimatorModelInfo modelInfo) {
		this.modelInfo = modelInfo;
	}

	public double predict(final String[] words) {
		return getMatchedWordsCount(modelInfo.getPopularWords(), words) / words.length;
	}

	private double getMatchedWordsCount(HashSet<String> popularWords, String[] words) {
		double count = 0.0;
		for (String word : words) {
			if (popularWords.contains(word)) {
				count++;
			}
		}
		return count;
	}

	@Override
	public void transform(Map<String, Object> input) {
		String key = modelInfo.getInputKeys().iterator().next();
		String[] inp = (String[]) input.get(key);
		input.put(modelInfo.getOutputKeys().iterator().next(), predict(inp));
	}

	@Override
	public Set<String> getInputKeys() {
		return modelInfo.getInputKeys();
	}

	@Override
	public Set<String> getOutputKeys() {
		return modelInfo.getOutputKeys();
	}
}