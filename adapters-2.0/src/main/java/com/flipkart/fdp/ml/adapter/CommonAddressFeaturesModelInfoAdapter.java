package com.flipkart.fdp.ml.adapter;

import com.flipkart.fdp.ml.modelinfo.CommonAddressFeaturesModelInfo;
import org.apache.spark.ml.CommonAddressFeatures;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;


public class CommonAddressFeaturesModelInfoAdapter extends AbstractModelInfoAdapter<CommonAddressFeatures, CommonAddressFeaturesModelInfo> {
	@Override
	CommonAddressFeaturesModelInfo getModelInfo(CommonAddressFeatures from) {
		CommonAddressFeaturesModelInfo modelInfo = new CommonAddressFeaturesModelInfo();
		modelInfo.setFavourableStarts(new HashSet<>(Arrays.asList(from.favourableStartWords())));
		modelInfo.setUnFavourableStarts(new HashSet<>(Arrays.asList(from.unfavourableStartWords())));

		modelInfo.setSanitizedAddressParam(from.getInputCol());
		modelInfo.setMergedAddressParam(from.getRawInputCol());

		modelInfo.setNumWordsParam(from.getNumWordsParam());
		modelInfo.setNumCommasParam(from.getNumCommasParams());
		modelInfo.setNumericPresentParam(from.getNumericPresentParam());
		modelInfo.setAddressLengthParam(from.getAddressLengthParam());
		modelInfo.setFavouredStartColParam(from.getFavouredStartColParam());
		modelInfo.setUnfavouredStartColParam(from.getUnfavouredStartColParam());

		Set<String> inputKeys = new LinkedHashSet<>();
		inputKeys.add(from.getInputCol());
		inputKeys.add(from.getRawInputCol());
		modelInfo.setInputKeys(inputKeys);

		Set<String> outputKeys = new LinkedHashSet<>();
		outputKeys.add(from.getNumWordsParam());
		outputKeys.add(from.getNumCommasParams());
		outputKeys.add(from.getNumericPresentParam());
		outputKeys.add(from.getAddressLengthParam());
		outputKeys.add(from.getFavouredStartColParam());
		outputKeys.add(from.getUnfavouredStartColParam());
		modelInfo.setOutputKeys(outputKeys);

		return modelInfo;
	}

	@Override
	public Class<CommonAddressFeatures> getSource() {
		return CommonAddressFeatures.class;
	}

	@Override
	public Class<CommonAddressFeaturesModelInfo> getTarget() {
		return CommonAddressFeaturesModelInfo.class;
	}
}
