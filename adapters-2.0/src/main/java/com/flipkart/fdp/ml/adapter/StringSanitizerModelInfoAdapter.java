package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.StringSanitizerModelInfo;
import org.apache.spark.ml.StringSanitizer;

import java.util.LinkedHashSet;
import java.util.Set;

public class StringSanitizerModelInfoAdapter extends AbstractModelInfoAdapter<StringSanitizer, StringSanitizerModelInfo> {
	@Override
	StringSanitizerModelInfo getModelInfo(StringSanitizer from) {
		StringSanitizerModelInfo modelInfo = new StringSanitizerModelInfo();

		Set<String> inputKeys = new LinkedHashSet<>();
		inputKeys.add(from.getInputCol());
		modelInfo.setInputKeys(inputKeys);

		Set<String> outputKeys = new LinkedHashSet<>();
		outputKeys.add(from.getOutputCol());
		modelInfo.setOutputKeys(outputKeys);
		return modelInfo;
	}

	@Override
	public Class<StringSanitizer> getSource() {
		return StringSanitizer.class;
	}

	@Override
	public Class<StringSanitizerModelInfo> getTarget() {
		return StringSanitizerModelInfo.class;
	}
}
