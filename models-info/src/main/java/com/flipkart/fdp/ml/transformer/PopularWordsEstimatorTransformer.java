package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.PopularWordsEstimatorModelInfo;

import java.util.*;

public class PopularWordsEstimatorTransformer implements Transformer {
	private PopularWordsEstimatorModelInfo modelInfo;

	public PopularWordsEstimatorTransformer(final PopularWordsEstimatorModelInfo modelInfo) {
		this.modelInfo = modelInfo;
	}

	public double predict(final String[] words) {
		List<String> matchedWords = getMatchedWords(modelInfo.getPopularWords(), words);
		return matchedWords.size() * 1.0 / words.length;
	}

	private List<String> getMatchedWords(HashSet<String> popularWords, String[] words) {
		List<String> matchedWords = new ArrayList<>();
		for (String word : words) {
			if (popularWords.contains(word)) {
				matchedWords.add(word);
			}
		}
		return matchedWords;
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