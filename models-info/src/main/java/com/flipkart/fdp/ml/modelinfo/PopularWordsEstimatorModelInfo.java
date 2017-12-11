package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.PopularWordsEstimatorTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;

import java.util.HashSet;

public class PopularWordsEstimatorModelInfo extends AbstractModelInfo {
	private HashSet<String> popularWords;

	@Override
	public Transformer getTransformer() {
		return new PopularWordsEstimatorTransformer(this);
	}

	public HashSet<String> getPopularWords() {
		return popularWords;
	}

	public void setPopularWords(HashSet<String> popularWords) {
		this.popularWords = popularWords;
	}
}