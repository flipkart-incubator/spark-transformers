package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.StringMergeModelInfo;
import org.apache.spark.ml.StringMerge;

import java.util.LinkedHashSet;
import java.util.Set;

public class StringMergeInfoAdapter extends AbstractModelInfoAdapter<StringMerge, StringMergeModelInfo> {
	@Override
	StringMergeModelInfo getModelInfo(StringMerge from) {
		StringMergeModelInfo modelInfo = new StringMergeModelInfo();

		Set<String> inputKeys = new LinkedHashSet<>();
		inputKeys.add(from.getInputCol1());
		inputKeys.add(from.getInputCol2());
		modelInfo.setInputKeys(inputKeys);

		Set<String> outputKeys = new LinkedHashSet<>();
		outputKeys.add(from.getOutputCol());
		modelInfo.setOutputKeys(outputKeys);
		return modelInfo;
	}

	@Override
	public Class<StringMerge> getSource() {
		return StringMerge.class;
	}

	@Override
	public Class<StringMergeModelInfo> getTarget() {
		return StringMergeModelInfo.class;
	}
}
