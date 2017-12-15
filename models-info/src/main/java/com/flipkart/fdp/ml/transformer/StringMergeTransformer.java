package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.StringMergeModelInfo;

import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class StringMergeTransformer implements Transformer {
	private StringMergeModelInfo modelInfo;

	public StringMergeTransformer(final StringMergeModelInfo modelInfo) {
		this.modelInfo = modelInfo;
	}

	@Override
	public void transform(Map<String, Object> input) {
		Iterator<String> iterator = modelInfo.getInputKeys().iterator();

		String input1 = (String) input.get(iterator.next());
		String input2 = (String) input.get(iterator.next());
		input.put(modelInfo.getOutputKeys().iterator().next(), transformInput(input1, input2));
	}

	private String transformInput(String input1, String input2) {
		return (input1 + " " + input2).trim();
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
