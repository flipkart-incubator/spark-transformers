package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.StringSanitizerModelInfo;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class StringSanitizerTransformer implements Transformer {
	private StringSanitizerModelInfo modelInfo;

	public StringSanitizerTransformer(StringSanitizerModelInfo modelInfo) {
		this.modelInfo = modelInfo;
	}

	@Override
	public void transform(Map<String, Object> input) {
		String key = modelInfo.getInputKeys().iterator().next();
		String inp = (String) input.get(key);
		input.put(modelInfo.getOutputKeys().iterator().next(), transformInput(inp));
	}

	private String[] transformInput(String input) {
		String s = input.toLowerCase()
				.replaceAll("\\P{Print}", " ")
				.replaceAll("[^0-9a-zA-Z ]", " ")
				.replaceAll("\\d{10}", " ")
				.replaceAll("\\d{6}", " ")
				.trim()
				.replaceAll("\\s+", " ");
		String[] split = s.split(" ");
		return split;
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
